"""
Decoder components for DiffSwinTr.

Implements patch expansion layers that are the inverse of timm's PatchMerging,
enabling the U-Net decoder to upsample features back to full resolution.
"""

import torch
import torch.nn as nn
from typing import Type


class PatchExpand(nn.Module):
    """
    Patch expanding layer for decoder upsampling.
    
    This is the inverse operation of PatchMerging:
    - PatchMerging: [B, H, W, C] → [B, H/2, W/2, 2C] (downsample)
    - PatchExpand:  [B, H, W, C] → [B, 2H, 2W, C/2] (upsample)
    
    Algorithm:
    1. Linear projection expands channels: C → 2C
    2. Reshape to double spatial dimensions: [B, H, W, 2C] → [B, 2H, 2W, C/2]
    3. Apply layer normalization
    
    Args:
        dim: Input channel dimension
        dim_scale: Spatial upsampling factor (default: 2)
        norm_layer: Normalization layer class
        
    Example:
        >>> expand = PatchExpand(dim=192, dim_scale=2)
        >>> x = torch.randn(2, 16, 16, 192)
        >>> out = expand(x)
        >>> print(out.shape)  # [2, 32, 32, 96]
    """
    
    def __init__(
        self, 
        dim: int, 
        dim_scale: int = 2, 
        norm_layer: Type[nn.Module] = nn.LayerNorm
    ):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.out_dim = dim // dim_scale
        
        # Linear expansion: C → (dim_scale^2) * (C // dim_scale)
        # For dim_scale=2: C → 4 * (C//2) = 2C
        self.expand = nn.Linear(dim, (dim_scale ** 2) * self.out_dim, bias=False)
        self.norm = norm_layer(self.out_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, H, W, C] in NHWC format
            
        Returns:
            Upsampled tensor [B, 2H, 2W, C/2] in NHWC format
        """
        B, H, W, C = x.shape
        
        # Expand channels: [B, H, W, C] → [B, H, W, dim_scale^2 * out_dim]
        x = self.expand(x)
        
        # Reshape to increase spatial resolution
        # [B, H, W, dim_scale^2 * out_dim] → [B, H, W, dim_scale, dim_scale, out_dim]
        x = x.view(B, H, W, self.dim_scale, self.dim_scale, self.out_dim)
        
        # Permute and reshape: [B, H, dim_scale, W, dim_scale, out_dim] → [B, H*dim_scale, W*dim_scale, out_dim]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * self.dim_scale, W * self.dim_scale, self.out_dim)
        
        # Normalize
        x = self.norm(x)
        
        return x


class FinalPatchExpand(nn.Module):
    """
    Final patch expansion to recover full image resolution.
    
    Converts [B, H/patch_size, W/patch_size, C] → [B, H, W, C]
    
    This is used as the last upsampling step before the classification head
    to go from patch resolution back to full image resolution.
    
    Args:
        dim: Input channel dimension
        patch_size: Original patch size used in PatchEmbed (typically 4)
        norm_layer: Normalization layer class
        
    Example:
        >>> expand = FinalPatchExpand(dim=96, patch_size=4)
        >>> x = torch.randn(2, 64, 64, 96)  # H/4, W/4 for 256px image
        >>> out = expand(x)
        >>> print(out.shape)  # [2, 256, 256, 96]
    """
    
    def __init__(
        self, 
        dim: int, 
        patch_size: int = 4, 
        norm_layer: Type[nn.Module] = nn.LayerNorm
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        
        # Expand channels to cover patch_size × patch_size spatial expansion
        self.expand = nn.Linear(dim, (patch_size ** 2) * dim, bias=False)
        self.norm = norm_layer(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, H, W, C]
            
        Returns:
            Upsampled tensor [B, H*patch_size, W*patch_size, C]
        """
        B, H, W, C = x.shape
        p = self.patch_size
        
        # Expand channels: [B, H, W, C] → [B, H, W, p*p*C]
        x = self.expand(x)
        
        # Reshape: [B, H, W, p, p, C]
        x = x.view(B, H, W, p, p, C)
        
        # Permute and reshape: [B, H, p, W, p, C] → [B, H*p, W*p, C]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * p, W * p, C)
        
        # Normalize
        x = self.norm(x)
        
        return x
