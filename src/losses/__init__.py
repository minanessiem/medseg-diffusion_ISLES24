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
from .discriminative_deep_supervision import (
    DiscriminativeDeepSupervisionResult,
    DiscriminativeTermSpec,
    SupervisionPlan,
    compute_discriminative_deep_supervision_loss,
    normalize_discriminative_head_outputs,
    resolve_discriminative_supervision_plan,
    resolve_discriminative_terms,
)

__all__ = [
    'DiceLoss',
    'DiceLossPerSample',
    'BCELoss',
    'CombinedSegmentationLoss',
    'DiscriminativeDeepSupervisionResult',
    'DiscriminativeTermSpec',
    'SupervisionPlan',
    'compute_discriminative_deep_supervision_loss',
    'normalize_discriminative_head_outputs',
    'resolve_discriminative_supervision_plan',
    'resolve_discriminative_terms',
]

