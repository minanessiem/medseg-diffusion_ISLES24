"""
DiffSwinTr - Swin Transformer U-Net for Diffusion-based Segmentation.

This package implements the DiffSwinTr architecture for medical image
segmentation using diffusion models with a Swin Transformer backbone.

Components:
    - DiffSwinTrAdapter: Pipeline adapter (main entry point)
    - SwinUNet: Core U-Net architecture
    - TimestepEmbedder: Sinusoidal time embeddings (from DiT)
    - ConditionalEncoderModule: CNN feature extractor for conditioning
    - PatchExpand: Decoder upsampling layer

Usage:
    # Via model factory (recommended)
    model = build_model(cfg)  # with cfg.model.architecture = "diffswintr"
    
    # Direct instantiation
    from src.models.DiffSwinTr import DiffSwinTrAdapter
    model = DiffSwinTrAdapter(cfg)

Key Features:
    - Swin Transformer backbone with window attention
    - U-Net encoder-decoder with skip connections
    - AdaLN-style time conditioning
    - Optional Conditional Encoder Module (CEM) for MRI conditioning
    - Compatible with existing diffusion adapter and training loop

Configuration:
    See configs/model/diffswintr_*.yaml for configuration options.

References:
    - DiffSwinTr paper: "A Diffusion Model with Swin Transformer for Brain Tumor Segmentation"
    - Swin Transformer: https://arxiv.org/abs/2103.14030
    - DiT: https://arxiv.org/abs/2212.09748
"""

from .model_adapter import DiffSwinTrAdapter
from .swin_unet import SwinUNet
from .time_conditioning import TimestepEmbedder, modulate
from .conditional_encoder import ConditionalEncoderModule, FeatureExtractionModule
from .decoder import PatchExpand, FinalPatchExpand

__all__ = [
    'DiffSwinTrAdapter',
    'SwinUNet',
    'TimestepEmbedder',
    'modulate',
    'ConditionalEncoderModule',
    'FeatureExtractionModule',
    'PatchExpand',
    'FinalPatchExpand',
]
