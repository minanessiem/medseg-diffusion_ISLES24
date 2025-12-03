"""
SwinUNet - Swin Transformer U-Net for diffusion-based segmentation.

This module assembles the core SwinUNet architecture using:
- timm's Swin Transformer components (encoder stages)
- Custom decoder with PatchExpand
- Time conditioning via AdaLN
- Optional CEM feature fusion
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

from timm.models.swin_transformer import SwinTransformerStage
from timm.layers import PatchEmbed, trunc_normal_

from .time_conditioning import TimestepEmbedder
from .decoder import PatchExpand, FinalPatchExpand
from .conditional_encoder import ConditionalEncoderModule
from .utils import fuse_cem_features


class SwinUNet(nn.Module):
    """
    Swin Transformer U-Net for diffusion-based segmentation.
    
    Architecture overview:
    - Patch embedding to convert image to patch tokens
    - 4-stage encoder using Swin Transformer blocks
    - 3-stage decoder with PatchExpand upsampling
    - Skip connections from encoder to decoder
    - Time conditioning at each stage
    - Optional CEM feature fusion
    
    The model operates in NHWC format internally (timm convention) and
    converts to NCHW for input/output (PyTorch convention).
    
    Args:
        img_size: Input image size (assumes square images)
        patch_size: Patch embedding size
        in_chans: Input channels (image_channels + mask_channels)
        out_chans: Output channels (typically 1 for noise prediction)
        embed_dim: Base embedding dimension (C in paper, typically 96)
        depths: Number of Swin blocks per encoder stage [s1, s2, s3, s4]
        num_heads: Number of attention heads per stage [s1, s2, s3, s4]
        window_size: Attention window size
        mlp_ratio: MLP hidden dimension expansion ratio
        time_embed_dim: Time embedding dimension
        cem_in_channels: Number of CEM input channels (image modalities only)
        cem_enabled: Whether to use Conditional Encoder Module
        cem_fusion_mode: How to fuse CEM features ("add" or "concat")
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        drop_path_rate: Stochastic depth rate
        
    Example:
        >>> model = SwinUNet(
        ...     img_size=256,
        ...     in_chans=3,
        ...     out_chans=1,
        ...     embed_dim=96,
        ...     depths=[2, 2, 6, 2],
        ...     num_heads=[3, 6, 12, 24],
        ... )
        >>> x = torch.randn(2, 3, 256, 256)
        >>> t_emb = torch.randn(2, 256)
        >>> out = model(x, t_emb)
        >>> print(out.shape)  # [2, 1, 256, 256]
    """
    
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 4,
        in_chans: int = 3,
        out_chans: int = 1,
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        time_embed_dim: int = 256,
        cem_in_channels: int = 2,
        cem_enabled: bool = True,
        cem_fusion_mode: str = "add",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        
        # Store configuration
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_stages = len(depths)
        self.cem_enabled = cem_enabled
        self.cem_fusion_mode = cem_fusion_mode
        self.out_chans = out_chans
        
        # Calculate dimensions at each stage
        # Stage dims: [C, 2C, 4C, 8C] = [96, 192, 384, 768] for embed_dim=96
        self.stage_dims = [embed_dim * (2 ** i) for i in range(self.num_stages)]
        
        # Patch grid size after patch embedding
        self.patch_grid = img_size // patch_size  # 256/4 = 64
        
        # ==================== Patch Embedding ====================
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm,
            output_fmt='NHWC',
        )
        
        # ==================== Conditional Encoder Module ====================
        if cem_enabled:
            self.cem = ConditionalEncoderModule(
                in_channels=cem_in_channels,
                embed_dim=embed_dim,
            )
        else:
            self.cem = None
        
        # ==================== Stochastic Depth ====================
        # Calculate drop path rates for each block (linearly increasing)
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        
        # ==================== Encoder Stages ====================
        self.encoder_stages = nn.ModuleList()
        self.encoder_time_mlps = nn.ModuleList()
        
        for i in range(self.num_stages):
            # For the first stage, no downsampling and input_dim = embed_dim
            # For subsequent stages, downsample and input_dim = stage_dims[i-1]
            if i == 0:
                input_dim = embed_dim
                input_resolution = (self.patch_grid, self.patch_grid)
                downsample = False
            else:
                input_dim = self.stage_dims[i - 1]
                input_resolution = (
                    self.patch_grid // (2 ** (i - 1)),
                    self.patch_grid // (2 ** (i - 1))
                )
                downsample = True
            
            output_dim = self.stage_dims[i]
            
            # Extract drop path rates for this stage
            stage_start = sum(depths[:i])
            stage_end = sum(depths[:i + 1])
            stage_dpr = dpr[stage_start:stage_end]
            
            # Encoder stage with downsampling (except first stage)
            stage = SwinTransformerStage(
                dim=input_dim,
                out_dim=output_dim,
                input_resolution=input_resolution,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                downsample=downsample,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=stage_dpr,
            )
            self.encoder_stages.append(stage)
            
            # Time conditioning MLP for this stage
            time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embed_dim, 2 * output_dim),
            )
            self.encoder_time_mlps.append(time_mlp)
        
        # ==================== Decoder Stages ====================
        self.decoder_stages = nn.ModuleList()
        self.decoder_time_mlps = nn.ModuleList()
        self.patch_expands = nn.ModuleList()
        
        # Decoder has num_stages - 1 stages (excluding bottleneck)
        # We go in reverse: from stage 3 down to stage 0
        for i in range(self.num_stages - 2, -1, -1):
            decoder_dim = self.stage_dims[i]
            input_dim = self.stage_dims[i + 1]
            
            # Patch expand for 2× upsampling
            self.patch_expands.append(PatchExpand(dim=input_dim, dim_scale=2))
            
            # Decoder stage resolution
            stage_resolution = (
                self.patch_grid // (2 ** i),
                self.patch_grid // (2 ** i),
            )
            
            # Decoder Swin blocks (no downsampling)
            decoder_stage = SwinTransformerStage(
                dim=decoder_dim,
                out_dim=decoder_dim,
                input_resolution=stage_resolution,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                downsample=False,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=0.0,  # No drop path in decoder
            )
            self.decoder_stages.append(decoder_stage)
            
            # Time conditioning for decoder
            time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embed_dim, 2 * decoder_dim),
            )
            self.decoder_time_mlps.append(time_mlp)
        
        # ==================== Output Layers ====================
        # Final patch expand to full resolution
        self.final_expand = FinalPatchExpand(dim=embed_dim, patch_size=patch_size)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, out_chans)
        
        # ==================== Weight Initialization ====================
        self.apply(self._init_weights)
    
    def _init_weights(self, m: nn.Module) -> None:
        """Initialize weights."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def _apply_time_conditioning(
        self, 
        x: torch.Tensor, 
        time_emb: torch.Tensor, 
        time_mlp: nn.Module
    ) -> torch.Tensor:
        """
        Apply time conditioning via scale and shift.
        
        Args:
            x: Features [B, H, W, C]
            time_emb: Time embedding [B, time_embed_dim]
            time_mlp: MLP to produce scale and shift
            
        Returns:
            Conditioned features [B, H, W, C]
        """
        B, H, W, C = x.shape
        
        # Get scale and shift from time embedding
        time_params = time_mlp(time_emb)  # [B, 2*C]
        scale, shift = time_params.chunk(2, dim=-1)  # [B, C] each
        
        # Reshape for broadcasting
        scale = scale.view(B, 1, 1, C)
        shift = shift.view(B, 1, 1, C)
        
        # Apply: x * (1 + scale) + shift
        return x * (1 + scale) + shift
    
    def forward(
        self, 
        x: torch.Tensor, 
        time_emb: torch.Tensor, 
        cem_features: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward pass of SwinUNet.
        
        Args:
            x: Combined input [B, in_chans, H, W] (image + noisy mask concatenated)
            time_emb: Time embedding [B, time_embed_dim]
            cem_features: Optional pre-computed CEM features [f1, f2, f3, f4]
                         If None and cem_enabled, will be computed from x
                         
        Returns:
            Output [B, out_chans, H, W] (noise prediction)
        """
        # ==================== Patch Embedding ====================
        # [B, C, H, W] → [B, H/4, W/4, embed_dim]
        x = self.patch_embed(x)  # Output is NHWC
        
        # ==================== Encoder ====================
        skip_connections = []
        
        for i, (stage, time_mlp) in enumerate(
            zip(self.encoder_stages, self.encoder_time_mlps)
        ):
            # Apply Swin stage (includes downsampling for stages > 0)
            x = stage(x)
            
            # Fuse CEM features if available (AFTER stage processing)
            if self.cem_enabled and cem_features is not None:
                x = fuse_cem_features(x, cem_features[i], self.cem_fusion_mode)
            
            # Apply time conditioning
            x = self._apply_time_conditioning(x, time_emb, time_mlp)
            
            # Save skip connection (except for bottleneck)
            if i < self.num_stages - 1:
                skip_connections.append(x)
        
        # ==================== Decoder ====================
        # Reverse skip connections for decoder (latest first)
        skip_connections = skip_connections[::-1]
        
        for i, (expand, stage, time_mlp) in enumerate(
            zip(self.patch_expands, self.decoder_stages, self.decoder_time_mlps)
        ):
            # Upsample
            x = expand(x)
            
            # Add skip connection
            x = x + skip_connections[i]
            
            # Apply Swin stage
            x = stage(x)
            
            # Apply time conditioning
            x = self._apply_time_conditioning(x, time_emb, time_mlp)
        
        # ==================== Output ====================
        # Final expand to full resolution
        x = self.final_expand(x)  # [B, H, W, embed_dim]
        
        # Project to output channels
        x = self.output_proj(x)  # [B, H, W, out_chans]
        
        # Convert to NCHW
        x = x.permute(0, 3, 1, 2)  # [B, out_chans, H, W]
        
        return x
