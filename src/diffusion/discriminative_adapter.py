"""
Discriminative segmentation adapter implementing the Diffusion interface.

This adapter enables training discriminative models (e.g., SwinUNETR) through
the existing diffusion training pipeline by implementing compatible interfaces.
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from .diffusion import Diffusion
from ..losses.discriminative_deep_supervision import (
    compute_discriminative_deep_supervision_loss,
    normalize_discriminative_head_outputs,
    resolve_discriminative_terms,
)


class DiscriminativeAdapter(Diffusion):
    """
    Adapter for discriminative segmentation models.
    
    Implements the Diffusion interface to allow discriminative models to be
    trained using the existing step_based_train loop without modification.
    
    Key differences from diffusion:
    - No noise, no timesteps, no iterative denoising
    - Model input: modalities only (not concatenated with mask)
    - Model output: segmentation logits (not noise)
    - Loss: configurable terms on logits and/or derived probabilities
    - Public sampling/logging APIs return probabilities for metric compatibility
    
    Args:
        model (nn.Module): Discriminative segmentation model
        cfg (DictConfig): Hydra configuration object
        device (torch.device, optional): Target device
    """
    
    def __init__(self, model: nn.Module, cfg: DictConfig, device=None):
        super().__init__(model, cfg, device)

        # For logger compatibility (expects num_timesteps for quartile logging)
        self.num_timesteps = 1

        # Parse and cache discriminative config as plain dict for fast forward calls.
        self.discriminative_cfg = OmegaConf.to_container(
            cfg.loss.discriminative,
            resolve=True,
        )
        if not isinstance(self.discriminative_cfg, dict):
            raise ValueError(
                "Expected cfg.loss.discriminative to resolve to a mapping."
            )

        # Fail-fast config validation at adapter init.
        resolved_terms = resolve_discriminative_terms(self.discriminative_cfg)

        deep_cfg = self.discriminative_cfg.get("deep_supervision", {})
        if not isinstance(deep_cfg, dict):
            deep_cfg = {}
        self.deep_supervision_enabled = bool(deep_cfg.get("enabled", False))
        self.final_head = int(deep_cfg.get("final_head", 0))
        self.head_parser = str(deep_cfg.get("head_parser", "auto"))
        self.spatial_dims = self._resolve_spatial_dims(model=model, cfg=cfg)

        # Trainer debug logging reads this field from diffusion adapters.
        self.last_loss_components = {}

        print("[DiscriminativeAdapter] Initialized:")
        print(f"  Deep supervision enabled: {self.deep_supervision_enabled}")
        print(f"  Final head index: {self.final_head}")
        print(f"  Head parser: {self.head_parser}")
        print(f"  Spatial dims: {self.spatial_dims}")
        print(f"  Loss terms: {[term.loss_key for term in resolved_terms]}")

    def _unwrap_model(self, model: nn.Module) -> nn.Module:
        """
        Return base model when wrapped in DP/DDP containers.
        """
        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return model.module
        return model

    def _parse_spatial_dims_token(self, value):
        """
        Parse supported 2D/3D spatial dims token.
        """
        if value is None:
            return None
        token = str(value).strip().lower()
        if token.endswith("d"):
            token = token[:-1]
        if token in {"2", "3"}:
            return int(token)
        return None

    def _resolve_spatial_dims(self, model: nn.Module, cfg: DictConfig):
        """
        Resolve spatial dims from model metadata first, then config fallback.

        Why:
            DataParallel/DistributedDataParallel wrappers do not expose custom
            module attributes (e.g., spatial_dims), so we must unwrap before
            reading model metadata.
        """
        base_model = self._unwrap_model(model)
        model_dims = self._parse_spatial_dims_token(getattr(base_model, "spatial_dims", None))
        if model_dims is not None:
            return model_dims

        cfg_model_dims = self._parse_spatial_dims_token(
            OmegaConf.select(cfg, "model.spatial_dims", default=None)
        )
        if cfg_model_dims is not None:
            return cfg_model_dims

        cfg_data_dims = self._parse_spatial_dims_token(
            OmegaConf.select(cfg, "data_mode.dim", default=None)
        )
        return cfg_data_dims

    def _resolve_inference_head_parser(self, model_output) -> str:
        """
        Resolve parser for inference-time outputs.

        DynUNet deep supervision returns stacked heads only in training mode.
        In eval mode it returns a single tensor, so we must detect parser by
        output rank rather than blindly trusting configured head_parser.
        """
        if isinstance(model_output, (list, tuple)):
            return "list"

        if torch.is_tensor(model_output):
            if self.spatial_dims in (2, 3):
                single_rank = int(self.spatial_dims) + 2   # [B,C,*spatial]
                stacked_rank = int(self.spatial_dims) + 3  # [B,S,C,*spatial]
                if model_output.dim() == single_rank:
                    return "single"
                if model_output.dim() == stacked_rank:
                    return "stacked"

            # Fallback when spatial dims are unavailable:
            # use permissive rank heuristics to avoid forcing "stacked" on
            # clear single-head outputs during eval.
            if self.head_parser == "stacked":
                # Ambiguous rank-5 case can be 3D single-head or 2D stacked.
                # Prefer single to avoid silently dropping channel axis by
                # misinterpreting [B,C,D,H,W] as [B,S,C,H,W].
                if model_output.dim() <= 5:
                    return "single"
                return "stacked"
            if self.head_parser == "auto":
                return "stacked" if model_output.dim() >= 6 else "single"
            return self.head_parser

        return "auto"

    def _logits_to_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert segmentation logits to probabilities for public inference paths.

        TODO: This assumes current discriminative tasks are binary single-channel
        segmentations. Future multi-class support should introduce an explicit
        activation policy instead of hard-coding sigmoid.
        """

        return torch.sigmoid(logits)

    def _validate_probability_output_rank(
        self,
        probability_mask: torch.Tensor,
        *,
        caller: str,
    ) -> None:
        if self.spatial_dims is None or not torch.is_tensor(probability_mask):
            return

        expected_dim = int(self.spatial_dims) + 2
        if probability_mask.dim() != expected_dim:
            raise ValueError(
                f"Unexpected {caller} output rank for discriminative inference. "
                f"Expected tensor dim {expected_dim} for spatial_dims={self.spatial_dims}, "
                f"got dim {probability_mask.dim()} with shape {tuple(probability_mask.shape)}. "
                "This often indicates a deep-supervision parser/config mismatch."
            )

    def _extract_final_probability_head(
        self,
        model_output,
        *,
        caller: str,
    ) -> torch.Tensor:
        """
        Parse model logits and return the configured final head as probabilities.

        Metrics, validation, and image logging consume probabilities even though
        loss computation may consume logits per configured loss term.
        """

        parser = self._resolve_inference_head_parser(model_output)
        logits_heads = normalize_discriminative_head_outputs(
            model_output=model_output,
            head_parser=parser,
        )
        if self.final_head not in logits_heads:
            raise ValueError(
                f"Configured final_head={self.final_head} is unavailable during {caller}. "
                f"Available heads: {sorted(logits_heads.keys())}"
            )
        probability_mask = self._logits_to_probabilities(logits_heads[self.final_head])
        self._validate_probability_output_rank(probability_mask, caller=caller)
        return probability_mask

    def forward(self, mask, conditioned_image, return_intermediates=False, global_step=0, *args, **kwargs):
        """
        Forward pass for discriminative training.
        
        Unlike diffusion models, this directly predicts the segmentation mask
        from the input modalities without any noise or timestep conditioning.
        
        Args:
            mask (torch.Tensor): Ground truth masks [B, 1, H, W] in [0, 1]
            conditioned_image (torch.Tensor): Input modalities [B, C, H, W]
            return_intermediates (bool): If True, return intermediate tensors
            global_step (int): Current training step (unused, for interface compat)
        
        Returns:
            loss (torch.Tensor): Scalar loss
            sample_mses (torch.Tensor): Dummy per-sample losses [B] (zeros)
            t (torch.Tensor): Dummy timesteps [B] (zeros)
            loss_components (dict): Individual loss values
        """
        mask = mask.to(self.device)
        conditioned_image = conditioned_image.to(self.device)
        batch_size = mask.shape[0]

        # Forward pass: model takes ONLY the modalities and returns logits.
        # No mask concatenation, no timestep conditioning.
        model_output = self.model(conditioned_image)

        loss_result = compute_discriminative_deep_supervision_loss(
            model_output=model_output,
            target=mask,
            discriminative_cfg=self.discriminative_cfg,
        )
        loss = loss_result.total_loss
        loss_components = loss_result.loss_components
        pred_mask = loss_result.final_prediction

        # Store loss components for trainer spike-debug parity.
        self.last_loss_components = loss_components

        # Dummy values for interface compatibility
        # These are unused for discriminative training but required for logger
        sample_mses = torch.zeros(batch_size, device=self.device)
        t = torch.zeros(batch_size, device=self.device, dtype=torch.long)

        if return_intermediates:
            intermediates = {
                'img': conditioned_image,
                'mask': mask,
                'pred': pred_mask,  # Probability-domain final head for visualization.
                # Discriminative-specific: no x_t, noise, noise_hat
            }
            return loss, sample_mses, t, intermediates

        return loss, sample_mses, t, loss_components

    def sample(self, conditioned_image, disable_tqdm=False):
        """
        Generate segmentation prediction.
        
        Unlike diffusion sampling which requires iterative denoising,
        discriminative inference is a single forward pass.
        
        Args:
            conditioned_image (torch.Tensor): Input modalities [B, C, H, W]
            disable_tqdm (bool): Unused (no iteration)
        
        Returns:
            torch.Tensor: Predicted masks [B, 1, H, W] in [0, 1]
        """
        conditioned_image = conditioned_image.to(self.device)

        with torch.no_grad():
            model_output = self.model(conditioned_image)
            pred_mask = self._extract_final_probability_head(
                model_output,
                caller="sample()",
            )

        return pred_mask

    def sample_with_snapshots(self, conditioned_image, snapshot_interval: int = None):
        """
        Generate prediction with snapshots (yields single result).
        
        For discriminative models, there's no iterative process, so we
        simply yield the final prediction at t=0.
        
        Args:
            conditioned_image (torch.Tensor): Input modalities [B, C, H, W]
            snapshot_interval (int): Unused (no iteration)
        
        Yields:
            tuple: (0, predicted_mask) - single result at t=0
        """
        conditioned_image = conditioned_image.to(self.device)

        with torch.no_grad():
            model_output = self.model(conditioned_image)
            pred_mask = self._extract_final_probability_head(
                model_output,
                caller="sample_with_snapshots()",
            )

        # Yield single snapshot at t=0 (final result)
        yield 0, pred_mask
