"""
Conditional Encoder Module (CEM) for DiffSwinTr.

The CEM extracts multi-scale CNN features from MRI images to condition
the Swin Transformer backbone, enhancing local feature perception.
"""

import torch
import torch.nn as nn
from typing import List


class FeatureExtractionModule(nn.Module):
    """
    Feature Extraction Module (FEM) from DiffSwinTr paper.
    
    Each FEM consists of two 3×3 convolutions with InstanceNorm and LeakyReLU.
    This design captures local features that complement the global attention
    in Swin Transformer blocks.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        
    Example:
        >>> fem = FeatureExtractionModule(96, 192)
        >>> x = torch.randn(2, 96, 32, 32)
        >>> out = fem(x)
        >>> print(out.shape)  # [2, 192, 32, 32]
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        # First conv block
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=3, padding=1, bias=False
        )
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.act1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # Second conv block
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 
            kernel_size=3, padding=1, bias=False
        )
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.act2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C_in, H, W]
            
        Returns:
            Output tensor [B, C_out, H, W]
        """
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x


class ConditionalEncoderModule(nn.Module):
    """
    Conditional Encoder Module (CEM) from DiffSwinTr paper.
    
    Extracts multi-scale CNN features from MRI images to condition the 
    Swin Transformer backbone. Features are extracted at 4 scales matching
    the encoder stage resolutions.
    
    Architecture:
        1. Initial 4×4 conv for patch embedding (matches Swin patch size)
        2. Four Feature Extraction Modules (FEMs) with progressive downsampling
        3. MaxPool 2×2 between FEMs for spatial reduction
    
    Channel progression (matching Swin encoder):
        - Stage 1: embed_dim (96)
        - Stage 2: 2 × embed_dim (192)
        - Stage 3: 4 × embed_dim (384)
        - Stage 4: 8 × embed_dim (768)
    
    Args:
        in_channels: Number of input image channels (e.g., 2 for ISLES24 modalities)
        embed_dim: Base embedding dimension (should match Swin's embed_dim)
        
    Example:
        >>> cem = ConditionalEncoderModule(in_channels=2, embed_dim=96)
        >>> x = torch.randn(2, 2, 256, 256)
        >>> features = cem(x)
        >>> for i, f in enumerate(features):
        ...     print(f"Stage {i+1}: {f.shape}")
        # Stage 1: [2, 96, 64, 64]
        # Stage 2: [2, 192, 32, 32]
        # Stage 3: [2, 384, 16, 16]
        # Stage 4: [2, 768, 8, 8]
    """
    
    def __init__(self, in_channels: int, embed_dim: int = 96):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Initial patch embedding via 4×4 conv (matches Swin patch size)
        # This gives 4× spatial downsampling to match PatchEmbed output
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=4, stride=4),
            nn.InstanceNorm2d(embed_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        
        # Feature Extraction Modules with progressive channel expansion
        # Channel dimensions: C → 2C → 4C → 8C (matching Swin encoder stages)
        self.fem1 = FeatureExtractionModule(embed_dim, embed_dim)            # 96 → 96
        self.fem2 = FeatureExtractionModule(embed_dim, embed_dim * 2)        # 96 → 192
        self.fem3 = FeatureExtractionModule(embed_dim * 2, embed_dim * 4)    # 192 → 384
        self.fem4 = FeatureExtractionModule(embed_dim * 4, embed_dim * 8)    # 384 → 768
        
        # 2× spatial downsampling between FEMs
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features from input image.
        
        Args:
            x: Input MRI image [B, C, H, W] where C is number of modalities
            
        Returns:
            List of 4 feature tensors at different scales, in NCHW format:
            - f1: [B, embed_dim, H/4, W/4]       (e.g., [B, 96, 64, 64])
            - f2: [B, 2*embed_dim, H/8, W/8]     (e.g., [B, 192, 32, 32])
            - f3: [B, 4*embed_dim, H/16, W/16]   (e.g., [B, 384, 16, 16])
            - f4: [B, 8*embed_dim, H/32, W/32]   (e.g., [B, 768, 8, 8])
        """
        # Initial embedding: [B, C, 256, 256] → [B, 96, 64, 64]
        x = self.patch_embed(x)
        
        # Stage 1: [B, 96, 64, 64] → [B, 96, 64, 64]
        f1 = self.fem1(x)
        
        # Stage 2: [B, 96, 64, 64] → pool → [B, 192, 32, 32]
        f2 = self.fem2(self.pool(f1))
        
        # Stage 3: [B, 192, 32, 32] → pool → [B, 384, 16, 16]
        f3 = self.fem3(self.pool(f2))
        
        # Stage 4 (Bottleneck): [B, 384, 16, 16] → pool → [B, 768, 8, 8]
        f4 = self.fem4(self.pool(f3))
        
        return [f1, f2, f3, f4]
