"""
Loss functions for multi-task segmentation training.

This module provides differentiable loss functions that are mathematically
aligned with their corresponding evaluation metrics in src.metrics.metrics.
"""

from .segmentation_losses import (
    DiceLoss,
    DiceLossPerSample,
    BCELoss,
    CombinedSegmentationLoss,
)

__all__ = [
    'DiceLoss',
    'DiceLossPerSample',
    'BCELoss',
    'CombinedSegmentationLoss',
]

