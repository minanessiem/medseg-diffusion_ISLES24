"""
MONAI SwinUNETR adapter for discriminative segmentation.

Wraps MONAI's SwinUNETR for use with the training pipeline,
exposing the required interface properties.
"""

import inspect

import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig

from monai.networks.nets import SwinUNETR


class SwinUNetRAdapter(nn.Module):
    """
    Wrapper for MONAI's SwinUNETR for discriminative segmentation (2D or 3D).
    
    Key differences from diffusion model adapters:
    - NO time conditioning (no timestep input)
    - NO mask concatenation (input is modalities only)
    - Direct mask prediction (not noise prediction)
    
    Args:
        cfg (DictConfig): Hydra configuration object
    
    Required config keys:
        cfg.model.image_size: Scalar edge length expanded to img_size by spatial_dims
        cfg.model.spatial_dims: 2D/3D token ("2d"/"3d") or integer (2/3)
        cfg.model.image_channels: Number of input channels (modalities)
        cfg.model.out_channels: Number of output channels (1 for binary seg)
        cfg.model.feature_size: SwinUNETR feature dimension
        cfg.model.depths: List of depths per stage
        cfg.model.num_heads: List of attention heads per stage
        cfg.model.drop_rate: Dropout rate
        cfg.model.attn_drop_rate: Attention dropout rate
    """
    
    def __init__(self, cfg: DictConfig):
        super().__init__()
        model_cfg = cfg.model

        spatial_dims_cfg = getattr(model_cfg, "spatial_dims", None)
        if spatial_dims_cfg is None and hasattr(cfg, "data_mode"):
            spatial_dims_cfg = getattr(cfg.data_mode, "dim", None)

        self.spatial_dims = self._parse_spatial_dims(spatial_dims_cfg)
        self.image_size = int(model_cfg.image_size)
        self.expanded_img_size = self._resolve_img_size(model_cfg.image_size, self.spatial_dims)

        # Store properties required by Diffusion base class
        self.image_channels = int(model_cfg.image_channels)
        self.mask_channels = int(model_cfg.out_channels)  # For interface compatibility
        self.output_channels = int(model_cfg.out_channels)

        # Parse list configs (handle both list and comma-separated string)
        depths = self._parse_list(model_cfg.depths)
        num_heads = self._parse_list(model_cfg.num_heads)

        # Initialize MONAI SwinUNETR with version-compatible kwargs.
        # MONAI <=1.3 expects `img_size`, while newer releases (e.g., 1.5.x)
        # removed it from the constructor.
        self.model = self._build_swinunetr(model_cfg, depths, num_heads)

        self._print_init_info(model_cfg, depths, num_heads)

    def _parse_list(self, value) -> list:
        """Parse list from config (handles list, tuple, ListConfig, or comma-separated string)."""
        if isinstance(value, (list, tuple, ListConfig)):
            return [int(x) for x in value]
        elif isinstance(value, str):
            return [int(x.strip()) for x in value.split(',')]
        else:
            raise ValueError(f"Cannot parse list from {type(value)}: {value}")

    def _build_swinunetr(self, model_cfg, depths, num_heads):
        """
        Build SwinUNETR with MONAI-version-aware constructor arguments.

        - MONAI 1.3.x: requires `img_size`.
        - MONAI 1.5.x: no `img_size` argument.
        """
        init_sig = inspect.signature(SwinUNETR.__init__)
        init_params = init_sig.parameters

        model_kwargs = {
            "in_channels": self.image_channels,
            "out_channels": self.output_channels,
            "feature_size": int(model_cfg.feature_size),
            "depths": tuple(depths),
            "num_heads": tuple(num_heads),
            "drop_rate": float(model_cfg.drop_rate),
            "attn_drop_rate": float(model_cfg.attn_drop_rate),
            "spatial_dims": self.spatial_dims,
            "use_checkpoint": False,  # Gradient checkpointing
        }

        if "img_size" in init_params:
            model_kwargs["img_size"] = self.expanded_img_size

        return SwinUNETR(**model_kwargs)

    def _parse_spatial_dims(self, value) -> int:
        """Parse supported 2D/3D spatial dims from config token."""
        if value is None:
            raise ValueError(
                "Missing spatial dims configuration. Expected cfg.model.spatial_dims "
                "or cfg.data_mode.dim."
            )

        token = str(value).strip().lower()
        if token.endswith("d"):
            token = token[:-1]
        if token not in {"2", "3"}:
            raise ValueError(
                f"Invalid spatial dims value '{value}'. Expected one of: 2, 3, '2d', '3d'."
            )
        return int(token)

    def _resolve_img_size(self, value, spatial_dims: int) -> tuple:
        """
        Expand scalar image_size to MONAI img_size tuple by spatial dims.

        Notes:
            `model.image_size` is intentionally scalar-only for now.
        """
        if isinstance(value, (list, tuple, ListConfig)):
            raise ValueError(
                "SwinUNetRAdapter expects model.image_size to be a scalar integer. "
                "List/tuple image_size is not supported."
            )

        image_size = int(value)
        if image_size <= 0:
            raise ValueError(f"model.image_size must be > 0, got {image_size}")
        return tuple([image_size] * spatial_dims)

    def _print_init_info(self, model_cfg, depths, num_heads):
        """Print initialization information."""
        print(f"[SwinUNetRAdapter] Initialized:")
        print(f"  Spatial dims: {self.spatial_dims}")
        print(f"  Image size (base): {self.image_size}")
        print(f"  Image size (expanded): {self.expanded_img_size}")
        print(f"  Input channels: {self.image_channels}")
        print(f"  Output channels: {self.output_channels}")
        print(f"  Feature size: {model_cfg.feature_size}")
        print(f"  Depths: {depths}")
        print(f"  Num heads: {num_heads}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for discriminative segmentation.

        Args:
            x: Input modalities [B, image_channels, *spatial_shape]

        Returns:
            Predicted segmentation [B, out_channels, *spatial_shape] in [0, 1]

        Note:
            Unlike diffusion models, this takes ONLY the input modalities.
            No timestep, no noisy mask - direct prediction.
        """
        logits = self.model(x)
        return torch.sigmoid(logits)
