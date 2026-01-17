"""
Discriminative segmentation adapter implementing the Diffusion interface.

This adapter enables training discriminative models (e.g., SwinUNETR) through
the existing diffusion training pipeline by implementing compatible interfaces.
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig

from .diffusion import Diffusion
from ..losses.segmentation_losses import DiceLoss, BCELoss


class DiscriminativeAdapter(Diffusion):
    """
    Adapter for discriminative segmentation models.
    
    Implements the Diffusion interface to allow discriminative models to be
    trained using the existing step_based_train loop without modification.
    
    Key differences from diffusion:
    - No noise, no timesteps, no iterative denoising
    - Model input: modalities only (not concatenated with mask)
    - Model output: predicted segmentation mask (not noise)
    - Loss: Dice and/or BCE on mask directly (configurable)
    
    Args:
        model (nn.Module): Discriminative segmentation model
        cfg (DictConfig): Hydra configuration object
        device (torch.device, optional): Target device
    """
    
    def __init__(self, model: nn.Module, cfg: DictConfig, device=None):
        super().__init__(model, cfg, device)
        
        # For logger compatibility (expects num_timesteps for quartile logging)
        self.num_timesteps = 1
        
        # Parse loss configuration (explicit - no defaults)
        loss_cfg = cfg.loss.discriminative
        
        # Initialize Dice loss (if enabled)
        self.dice_enabled = loss_cfg.dice.enabled
        if self.dice_enabled:
            self.dice_loss_fn = DiceLoss(
                smooth=loss_cfg.dice.smooth,
                apply_sigmoid=loss_cfg.dice.apply_sigmoid,
            )
            self.dice_weight = loss_cfg.dice.weight
        else:
            self.dice_loss_fn = None
            self.dice_weight = 0.0
        
        # Initialize BCE loss (if enabled)
        self.bce_enabled = loss_cfg.bce.enabled
        if self.bce_enabled:
            self.bce_loss_fn = BCELoss(
                pos_weight=loss_cfg.bce.pos_weight,
                apply_sigmoid=loss_cfg.bce.apply_sigmoid,
            )
            self.bce_weight = loss_cfg.bce.weight
        else:
            self.bce_loss_fn = None
            self.bce_weight = 0.0
        
        # Validate at least one loss is enabled
        if not self.dice_enabled and not self.bce_enabled:
            raise ValueError(
                "At least one discriminative loss must be enabled. "
                "Set loss.discriminative.dice.enabled=true and/or "
                "loss.discriminative.bce.enabled=true"
            )
        
        print("[DiscriminativeAdapter] Initialized:")
        print(f"  Dice: enabled={self.dice_enabled}, weight={self.dice_weight}")
        print(f"  BCE: enabled={self.bce_enabled}, weight={self.bce_weight}")
    
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
        
        # Forward pass: model takes ONLY the modalities
        # No mask concatenation, no timestep conditioning
        pred_mask = self.model(conditioned_image)
        
        # Compute enabled losses and build components dict
        loss = torch.tensor(0.0, device=self.device)
        loss_components = {}
        
        if self.dice_enabled:
            dice_loss = self.dice_loss_fn(pred_mask, mask)
            loss = loss + self.dice_weight * dice_loss
            loss_components['dice'] = dice_loss.item()
        
        if self.bce_enabled:
            bce_loss = self.bce_loss_fn(pred_mask, mask)
            loss = loss + self.bce_weight * bce_loss
            loss_components['bce'] = bce_loss.item()
        
        loss_components['total'] = loss.item()
        
        # Dummy values for interface compatibility
        # These are unused for discriminative training but required for logger
        sample_mses = torch.zeros(batch_size, device=self.device)
        t = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        
        if return_intermediates:
            intermediates = {
                'img': conditioned_image,
                'mask': mask,
                'pred': pred_mask,
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
            pred_mask = self.model(conditioned_image)
        
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
            pred_mask = self.model(conditioned_image)
        
        # Yield single snapshot at t=0 (final result)
        yield 0, pred_mask
