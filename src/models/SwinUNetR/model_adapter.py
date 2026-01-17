"""
MONAI SwinUNETR adapter for discriminative segmentation.

Wraps MONAI's SwinUNETR for use with the training pipeline,
exposing the required interface properties.
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig, ListConfig

from monai.networks.nets import SwinUNETR


class SwinUNetRAdapter(nn.Module):
    """
    Wrapper for MONAI's SwinUNETR for discriminative 2D segmentation.
    
    Key differences from diffusion model adapters:
    - NO time conditioning (no timestep input)
    - NO mask concatenation (input is modalities only)
    - Direct mask prediction (not noise prediction)
    
    Args:
        cfg (DictConfig): Hydra configuration object
    
    Required config keys:
        cfg.model.image_size: Input image size (assumes square)
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
        
        # Store properties required by Diffusion base class
        self.image_size = model_cfg.image_size
        self.image_channels = model_cfg.image_channels
        self.mask_channels = model_cfg.out_channels  # For interface compatibility
        self.output_channels = model_cfg.out_channels
        
        # Parse list configs (handle both list and comma-separated string)
        depths = self._parse_list(model_cfg.depths)
        num_heads = self._parse_list(model_cfg.num_heads)
        
        # Initialize MONAI SwinUNETR
        self.model = SwinUNETR(
            img_size=(model_cfg.image_size, model_cfg.image_size),
            in_channels=model_cfg.image_channels,
            out_channels=model_cfg.out_channels,
            feature_size=model_cfg.feature_size,
            depths=tuple(depths),
            num_heads=tuple(num_heads),
            drop_rate=model_cfg.drop_rate,
            attn_drop_rate=model_cfg.attn_drop_rate,
            spatial_dims=2,  # 2D segmentation
            use_checkpoint=False,  # Gradient checkpointing
        )
        
        self._print_init_info(model_cfg, depths, num_heads)
    
    def _parse_list(self, value) -> list:
        """Parse list from config (handles list, tuple, ListConfig, or comma-separated string)."""
        if isinstance(value, (list, tuple, ListConfig)):
            return [int(x) for x in value]
        elif isinstance(value, str):
            return [int(x.strip()) for x in value.split(',')]
        else:
            raise ValueError(f"Cannot parse list from {type(value)}: {value}")
    
    def _print_init_info(self, model_cfg, depths, num_heads):
        """Print initialization information."""
        print(f"[SwinUNetRAdapter] Initialized:")
        print(f"  Image size: {self.image_size}")
        print(f"  Input channels: {self.image_channels}")
        print(f"  Output channels: {self.output_channels}")
        print(f"  Feature size: {model_cfg.feature_size}")
        print(f"  Depths: {depths}")
        print(f"  Num heads: {num_heads}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for discriminative segmentation.
        
        Args:
            x: Input modalities [B, image_channels, H, W]
        
        Returns:
            Predicted segmentation [B, out_channels, H, W] in [0, 1]
        
        Note:
            Unlike diffusion models, this takes ONLY the input modalities.
            No timestep, no noisy mask - direct prediction.
        """
        logits = self.model(x)
        return torch.sigmoid(logits)
