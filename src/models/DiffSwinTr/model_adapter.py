"""
DiffSwinTr adapter for pipeline integration.

This adapter wraps SwinUNet to match the interface expected by the
existing diffusion training pipeline (same pattern as ORGMedSegDiffAdapter).
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import List, Union

from .swin_unet import SwinUNet
from .time_conditioning import TimestepEmbedder
from .conditional_encoder import ConditionalEncoderModule


class DiffSwinTrAdapter(nn.Module):
    """
    Adapter wrapping SwinUNet for pipeline compatibility.
    
    This adapter follows the same pattern as ORGMedSegDiffAdapter:
    1. Translates Hydra config to model parameters
    2. Handles input format conversion
    3. Exposes standard pipeline interface
    
    Pipeline Interface:
        Input: x [B, 1, H, W], timesteps [B], conditioned_image [B, C, H, W]
        Output: noise_prediction [B, 1, H, W]
    
    Args:
        cfg (DictConfig): Hydra configuration object with cfg.model section
        
    Attributes:
        image_size (int): Input image size
        mask_channels (int): Number of mask channels
        image_channels (int): Number of image channels
        output_channels (int): Number of output channels
        produces_calibration (bool): False - no auxiliary output
        
    Example:
        >>> cfg = OmegaConf.load("configs/model/diffswintr_b.yaml")
        >>> model = DiffSwinTrAdapter(cfg)
        >>> x = torch.randn(2, 1, 256, 256)
        >>> t = torch.randint(0, 1000, (2,))
        >>> img = torch.randn(2, 2, 256, 256)
        >>> out = model(x, t, img)
        >>> print(out.shape)  # [2, 1, 256, 256]
    """
    
    def __init__(self, cfg: DictConfig):
        super().__init__()
        
        self.cfg = cfg
        model_cfg = cfg.model
        
        # Extract and store configuration
        self.image_size = model_cfg.image_size
        self.mask_channels = model_cfg.mask_channels
        self.image_channels = model_cfg.image_channels
        self.output_channels = model_cfg.out_channels
        
        # Combined input channels for SwinUNet
        in_chans = self.mask_channels + self.image_channels
        
        # Parse list configurations
        depths = self._parse_list(model_cfg.depths)
        num_heads = self._parse_list(model_cfg.num_heads)
        
        # Get optional configuration with defaults
        time_embed_dim = model_cfg.get('time_embed_dim', 256)
        cem_enabled = model_cfg.get('cem_enabled', True)
        cem_fusion_mode = model_cfg.get('cem_fusion_mode', 'add')
        drop_rate = model_cfg.get('drop_rate', 0.0)
        attn_drop_rate = model_cfg.get('attn_drop_rate', 0.0)
        drop_path_rate = model_cfg.get('drop_path_rate', 0.1)
        
        # ==================== Build Components ====================
        
        # Time embedder
        self.time_embedder = TimestepEmbedder(
            hidden_size=time_embed_dim,
            frequency_embedding_size=256,
        )
        
        # Conditional Encoder Module (if enabled)
        self.cem_enabled = cem_enabled
        if cem_enabled:
            self.cem = ConditionalEncoderModule(
                in_channels=self.image_channels,
                embed_dim=model_cfg.embed_dim,
            )
        else:
            self.cem = None
        
        # Main SwinUNet (CEM disabled here, we handle it in adapter)
        self.swin_unet = SwinUNet(
            img_size=self.image_size,
            patch_size=model_cfg.patch_size,
            in_chans=in_chans,
            out_chans=self.output_channels,
            embed_dim=model_cfg.embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=model_cfg.window_size,
            mlp_ratio=model_cfg.mlp_ratio,
            time_embed_dim=time_embed_dim,
            cem_in_channels=self.image_channels,
            cem_enabled=False,  # We pass CEM features from adapter
            cem_fusion_mode=cem_fusion_mode,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )
        
        # Pipeline compatibility flag
        self.produces_calibration = False
        
        # Print initialization info
        self._print_init_info(model_cfg, depths, num_heads, cem_enabled)
    
    def _parse_list(self, value: Union[str, List, tuple]) -> List[int]:
        """Parse comma-separated string or list to list of ints."""
        if isinstance(value, str):
            return [int(x.strip()) for x in value.split(',')]
        elif isinstance(value, (list, tuple)):
            return [int(x) for x in value]
        else:
            # Handle OmegaConf ListConfig or DictConfig
            try:
                return [int(x) for x in value]
            except (TypeError, ValueError):
                raise ValueError(f"Cannot parse list from {type(value)}: {value}")
    
    def _print_init_info(self, model_cfg, depths, num_heads, cem_enabled):
        """Print initialization information."""
        in_chans = self.mask_channels + self.image_channels
        print(f"[DiffSwinTrAdapter] Initialized:")
        print(f"  Image size: {self.image_size}")
        print(f"  Input channels: {in_chans} "
              f"(image: {self.image_channels}, mask: {self.mask_channels})")
        print(f"  Output channels: {self.output_channels}")
        print(f"  Embed dim: {model_cfg.embed_dim}")
        print(f"  Depths: {depths}")
        print(f"  Num heads: {num_heads}")
        print(f"  Window size: {model_cfg.window_size}")
        print(f"  CEM enabled: {cem_enabled}")
    
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        conditioned_image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with pipeline interface.
        
        This method matches the interface expected by ConditionalModelWrapper
        and GaussianDiffusionAdapter.
        
        Args:
            x: Noisy mask [B, mask_channels, H, W]
            timesteps: Diffusion timesteps [B]
            conditioned_image: Conditioning image [B, image_channels, H, W]
            
        Returns:
            Noise prediction [B, output_channels, H, W]
        """
        # Compute time embedding
        time_emb = self.time_embedder(timesteps)
        
        # Compute CEM features if enabled
        if self.cem is not None:
            cem_features = self.cem(conditioned_image)
        else:
            cem_features = None
        
        # Concatenate conditioning image and noisy mask
        # Convention: [image, mask] to match other implementations
        combined = torch.cat([conditioned_image, x], dim=1)
        
        # Forward through SwinUNet
        noise_pred = self.swin_unet(combined, time_emb, cem_features)
        
        return noise_pred
