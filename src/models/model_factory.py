"""
Model factory for architecture instantiation.

Supports multiple architecture types via registry pattern.
"""

from typing import Callable, Dict
import torch.nn as nn
from omegaconf import DictConfig


# Architecture registry
_ARCHITECTURE_REGISTRY: Dict[str, Callable[[DictConfig], nn.Module]] = {}


def register_architecture(name: str):
    """
    Decorator to register architecture builders.
    
    Usage:
        @register_architecture('medsegdiff')
        def build_medsegdiff(cfg):
            from .MedSegDiff import Unet
            return Unet(cfg)
    
    Args:
        name: Architecture name (will be lowercased for matching)
    
    Returns:
        Decorator function that registers the builder
    """
    def decorator(builder_fn: Callable[[DictConfig], nn.Module]):
        _ARCHITECTURE_REGISTRY[name.lower()] = builder_fn
        return builder_fn
    return decorator


def build_model(cfg: DictConfig) -> nn.Module:
    """
    Build model based on cfg.model.architecture.
    
    Args:
        cfg: Hydra configuration with cfg.model.architecture specified
        
    Returns:
        Initialized model (nn.Module)
        
    Raises:
        ValueError: If architecture not registered
        
    Example:
        >>> from omegaconf import OmegaConf
        >>> from src.models import build_model
        >>> cfg = OmegaConf.load('configs/model/medsegdiff_32.yaml')
        >>> model = build_model(cfg)
        >>> print(type(model).__name__)  # 'Unet'
    """
    arch_name = cfg.model.architecture.lower()
    
    if arch_name not in _ARCHITECTURE_REGISTRY:
        available = list(_ARCHITECTURE_REGISTRY.keys())
        raise ValueError(
            f"Unknown architecture: '{arch_name}'. "
            f"Available architectures: {available}"
        )
    
    builder_fn = _ARCHITECTURE_REGISTRY[arch_name]
    model = builder_fn(cfg)
    
    print(f"Built model architecture: {arch_name}")
    return model


# ============================================================================
# Register Available Architectures
# ============================================================================

@register_architecture('medsegdiff')
def build_medsegdiff(cfg: DictConfig) -> nn.Module:
    """
    Build MedSegDiff (dual-stream) architecture.
    
    Paper: "MedSegDiff: Medical Image Segmentation with Diffusion Model"
    Architecture: Dual parallel encoders with dynamic fusion
    """
    from .MedSegDiff import Unet
    return Unet(cfg)


@register_architecture('org_medsegdiff')
def build_org_medsegdiff(cfg: DictConfig) -> nn.Module:
    """
    Build Official MedSegDiff architecture.
    
    Uses thin adapter around untouched official code.
    Supports both 'new' (UNetModel_newpreview) and 'v1' (UNetModel_v1preview) versions.
    
    Key features:
        - Highway network produces calibration output (auxiliary segmentation)
        - Calibration output has sigmoid applied internally
        - Returns tuple: (noise_prediction, calibration_output)
    
    Required config keys:
        cfg.model.architecture: "org_medsegdiff"
        cfg.model.version: "new" or "v1" (default: "new")
        cfg.model.image_size: int
        cfg.model.in_channels: int (image_channels + mask_channels)
        cfg.model.model_channels: int
        cfg.model.out_channels: int
    
    Optional config keys:
        cfg.model.channel_mult: str (comma-separated, e.g., "1,2,4,8")
        cfg.model.attention_resolutions: str (comma-separated resolutions)
        cfg.model.num_res_blocks: int (default: 2)
        cfg.model.num_heads: int (default: 4)
        cfg.model.highway.enabled: bool (default: True)
    """
    from .ORGMedSegDiff import ORGMedSegDiffAdapter
    return ORGMedSegDiffAdapter(cfg)


@register_architecture('diffswintr')
def build_diffswintr(cfg: DictConfig) -> nn.Module:
    """
    Build DiffSwinTr (Swin Transformer U-Net) architecture.
    
    Paper: "DiffSwinTr: A Diffusion Model with Swin Transformer 
           for Brain Tumor Segmentation"
    
    Key features:
        - Swin Transformer backbone with window attention
        - Conditional Encoder Module (CEM) for MRI conditioning
        - AdaLN-style time conditioning
        - U-Net encoder-decoder with skip connections
    
    Required config keys:
        cfg.model.architecture: "diffswintr"
        cfg.model.image_size: int
        cfg.model.patch_size: int
        cfg.model.embed_dim: int
        cfg.model.depths: list[int] or comma-separated string
        cfg.model.num_heads: list[int] or comma-separated string
        cfg.model.window_size: int
        cfg.model.mask_channels: int
        cfg.model.image_channels: int
        cfg.model.out_channels: int
    
    Optional config keys:
        cfg.model.mlp_ratio: float (default: 4.0)
        cfg.model.time_embed_dim: int (default: 256)
        cfg.model.cem_enabled: bool (default: True)
        cfg.model.cem_fusion_mode: str (default: "add")
        cfg.model.drop_rate: float (default: 0.0)
        cfg.model.attn_drop_rate: float (default: 0.0)
        cfg.model.drop_path_rate: float (default: 0.1)
    """
    from .DiffSwinTr import DiffSwinTrAdapter
    return DiffSwinTrAdapter(cfg)


@register_architecture('swinunetr')
def build_swinunetr(cfg: DictConfig) -> nn.Module:
    """
    Build MONAI SwinUNETR for discriminative segmentation.
    
    This is a discriminative model (no diffusion) that takes only
    the input modalities and directly predicts the segmentation mask.
    
    Required config keys:
        cfg.model.architecture: "swinunetr"
        cfg.model.image_size: int
        cfg.model.image_channels: int
        cfg.model.out_channels: int
        cfg.model.feature_size: int
        cfg.model.depths: list[int] or comma-separated string
        cfg.model.num_heads: list[int] or comma-separated string
        cfg.model.drop_rate: float
        cfg.model.attn_drop_rate: float
    """
    from .SwinUNetR import SwinUNetRAdapter
    return SwinUNetRAdapter(cfg)
