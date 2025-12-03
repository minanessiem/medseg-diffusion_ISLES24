"""
Utility functions for DiffSwinTr.
"""

import torch
import torch.nn as nn
from typing import Optional


def fuse_cem_features(
    encoder_features: torch.Tensor,
    cem_features: torch.Tensor,
    fusion_mode: str = "add",
) -> torch.Tensor:
    """
    Fuse CEM features into encoder features.
    
    The CEM produces features in NCHW format (standard CNN), while the
    Swin encoder uses NHWC format. This function handles the conversion
    and fusion.
    
    Args:
        encoder_features: Encoder output in NHWC format [B, H, W, C]
        cem_features: CEM output in NCHW format [B, C, H, W]
        fusion_mode: Fusion strategy - "add" or "concat"
            - "add": Element-wise addition (implemented)
            - "concat": Channel concatenation (STUBBED - not implemented)
            
    Returns:
        Fused features in NHWC format [B, H, W, C]
        
    Raises:
        NotImplementedError: If fusion_mode is "concat"
        ValueError: If fusion_mode is unknown
        
    Example:
        >>> enc_feat = torch.randn(2, 32, 32, 192)  # NHWC
        >>> cem_feat = torch.randn(2, 192, 32, 32)  # NCHW
        >>> fused = fuse_cem_features(enc_feat, cem_feat, "add")
        >>> print(fused.shape)  # [2, 32, 32, 192]
    """
    # Convert CEM features from NCHW to NHWC
    cem_nhwc = cem_features.permute(0, 2, 3, 1)  # [B, H, W, C]
    
    if fusion_mode == "add":
        # Simple element-wise addition
        return encoder_features + cem_nhwc
    
    elif fusion_mode == "concat":
        # STUBBED: Concatenation fusion is not implemented in v1.0
        # Future implementation would:
        # 1. Concatenate: [B, H, W, 2C]
        # 2. Project back to C channels via linear layer
        raise NotImplementedError(
            "Concatenation fusion is stubbed for future implementation. "
            "Please use fusion_mode='add' for now."
        )
    
    else:
        raise ValueError(
            f"Unknown fusion_mode: '{fusion_mode}'. "
            f"Supported modes: 'add', 'concat' (stubbed)"
        )


def convert_nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor from NHWC to NCHW format.
    
    Args:
        x: Tensor in NHWC format [B, H, W, C]
        
    Returns:
        Tensor in NCHW format [B, C, H, W]
        
    Example:
        >>> x_nhwc = torch.randn(2, 64, 64, 96)
        >>> x_nchw = convert_nhwc_to_nchw(x_nhwc)
        >>> print(x_nchw.shape)  # [2, 96, 64, 64]
    """
    return x.permute(0, 3, 1, 2)


def convert_nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor from NCHW to NHWC format.
    
    Args:
        x: Tensor in NCHW format [B, C, H, W]
        
    Returns:
        Tensor in NHWC format [B, H, W, C]
        
    Example:
        >>> x_nchw = torch.randn(2, 96, 64, 64)
        >>> x_nhwc = convert_nchw_to_nhwc(x_nchw)
        >>> print(x_nhwc.shape)  # [2, 64, 64, 96]
    """
    return x.permute(0, 2, 3, 1)
