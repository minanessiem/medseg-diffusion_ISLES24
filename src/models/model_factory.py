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


# Future architectures can be registered here:
# 
# @register_architecture('diffswintr')
# def build_diffswintr(cfg: DictConfig) -> nn.Module:
#     """Build DiffSwinTr (3D Swin Transformer) architecture."""
#     from .DiffSwinTr import SwinUNet
#     return SwinUNet(cfg)
# 
# @register_architecture('medsegdiff_highway')
# def build_medsegdiff_highway(cfg: DictConfig) -> nn.Module:
#     """Build MedSegDiff with highway network (original repo variant)."""
#     from .MedSegDiff_Highway import UNetModel_v1preview
#     return UNetModel_v1preview(cfg)

