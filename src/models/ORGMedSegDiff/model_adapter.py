"""
Adapter wrapping official MedSegDiff for pipeline compatibility.

This thin wrapper handles:
1. Config translation (Hydra -> official parameters)
2. Input format conversion (separate mask+image -> concatenated)
3. Output tuple preservation (noise_pred, calibration)

The adapter maintains the "graft, don't rewrite" philosophy by keeping
all adaptation at the interface boundary while leaving the official
model code untouched.
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import Tuple


class ORGMedSegDiffAdapter(nn.Module):
    """
    Thin adapter for official MedSegDiff models.
    
    Translates between our pipeline interface and official model interface:
    
    Pipeline Interface:
        Input: x [B, 1, H, W], timesteps [B], conditioned_image [B, C, H, W]
        Output: (noise_pred [B, out_ch, H, W], cal [B, 1, H, W])
    
    Official Interface:
        Input: x [B, C+1, H, W] (concatenated), timesteps [B]
        Output: (out, cal)
    
    Args:
        cfg (DictConfig): Hydra configuration object
    
    Example:
        >>> adapter = ORGMedSegDiffAdapter(cfg)
        >>> noise_pred, cal = adapter(mask, timesteps, conditioned_image)
    
    Attributes:
        produces_calibration (bool): Always True - signals that this model
            returns a tuple with calibration output for auxiliary loss.
    """
    
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        # Store config for reference
        self.cfg = cfg
        model_cfg = cfg.model
        
        # Translate config to official parameters
        params = self._translate_config(model_cfg)
        
        # Import and instantiate official model
        version = model_cfg.version
        if version == 'new':
            from .unet import UNetModel_newpreview
            self.model = UNetModel_newpreview(**params)
        elif version == 'v1':
            from .unet import UNetModel_v1preview
            self.model = UNetModel_v1preview(**params)
        else:
            raise ValueError(f"Unknown version: {version}. Use 'new' or 'v1'.")
        
        # Expose properties for pipeline compatibility
        self.image_size = model_cfg.image_size
        self.mask_channels = model_cfg.mask_channels
        self.image_channels = model_cfg.image_channels
        self.output_channels = model_cfg.out_channels
        
        # Compute in_channels for logging (same as in _translate_config)
        in_channels = self.mask_channels + self.image_channels
        
        # Flag for adapter to know this model produces calibration
        self.produces_calibration = True
        
        print(f"[ORGMedSegDiffAdapter] Initialized:")
        print(f"  Version: {version}")
        print(f"  Image size: {self.image_size}")
        print(f"  Input channels: {in_channels} "
              f"(image: {self.image_channels}, mask: {self.mask_channels})")
        print(f"  Output channels: {self.output_channels}")
        print(f"  Highway network: {model_cfg.highway.enabled}")
    
    def _translate_config(self, model_cfg: DictConfig) -> dict:
        """
        Translate Hydra config to official model parameters.
        
        Handles conversion of string-based config values (e.g., channel_mult
        as comma-separated string) to the formats expected by the official
        model constructors.
        
        Args:
            model_cfg: The model section of the Hydra config
            
        Returns:
            Dictionary of parameters for official model constructor
        """
        # Parse channel_mult string to tuple
        channel_mult = tuple(
            int(x.strip()) for x in model_cfg.channel_mult.split(',')
        )
        
        # Parse attention_resolutions string to tuple of downsample rates
        # Config specifies resolutions (e.g., "16,8"), model expects ds rates
        attention_ds = []
        for res in model_cfg.attention_resolutions.split(','):
            res_int = int(res.strip())
            if res_int > 0:
                attention_ds.append(model_cfg.image_size // res_int)
        
        # Compute in_channels from components (enables Hydra interpolation for image_channels)
        in_channels = model_cfg.mask_channels + model_cfg.image_channels
        
        return {
            'image_size': model_cfg.image_size,
            'in_channels': in_channels,
            'model_channels': model_cfg.model_channels,
            'out_channels': model_cfg.out_channels,
            'num_res_blocks': model_cfg.num_res_blocks,
            'attention_resolutions': tuple(attention_ds),
            'dropout': model_cfg.dropout,
            'channel_mult': channel_mult,
            'num_heads': model_cfg.num_heads,
            'num_head_channels': model_cfg.num_head_channels,
            'num_heads_upsample': model_cfg.num_heads_upsample,
            'use_scale_shift_norm': model_cfg.use_scale_shift_norm,
            'resblock_updown': model_cfg.resblock_updown,
            'use_fp16': model_cfg.use_fp16,
            'use_checkpoint': model_cfg.use_checkpoint,
            'use_new_attention_order': model_cfg.use_new_attention_order,
            'high_way': model_cfg.highway.enabled,
        }
    
    def forward(
        self, 
        x: torch.Tensor, 
        timesteps: torch.Tensor, 
        conditioned_image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with interface translation.
        
        Concatenates the conditioning image and noisy mask in the order
        expected by the official model:
            combined = [conditioned_image, x]
        
        This matches the official convention where:
            c = h[:,:-1,...]  # extracts conditioning (all but last channel)
        
        Args:
            x: Noisy mask [B, 1, H, W]
            timesteps: Diffusion timesteps [B]
            conditioned_image: Conditioning image [B, C, H, W]
        
        Returns:
            Tuple[Tensor, Tensor]: (noise_prediction, calibration_output)
                - noise_prediction: [B, out_channels, H, W]
                - calibration_output: [B, 1, H, W] with sigmoid applied
        """
        # Official expects concatenated input: [conditioning_image, mask]
        # This matches the convention: c = h[:,:-1,...] extracts conditioning
        combined = torch.cat([conditioned_image, x], dim=1)
        
        # Call official model (returns tuple)
        out, cal = self.model(combined, timesteps)
        
        return out, cal
