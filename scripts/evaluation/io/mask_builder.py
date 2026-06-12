"""
Mask conversion and validation helpers for evaluation pipelines.

This module centralizes binary-mask handling so IO adapters and entrypoints
apply consistent thresholding rules.
"""

from typing import Optional

import torch
from torch import Tensor


def threshold_probabilities(prediction_prob: Tensor, threshold: float) -> Tensor:
    """Convert probability tensor to binary mask at threshold."""
    _validate_threshold(threshold)
    if prediction_prob is None:
        raise ValueError("prediction_prob must not be None.")
    return (prediction_prob > threshold).float()


def ensure_binary_mask(mask: Tensor) -> Tensor:
    """
    Normalize a mask tensor to strict binary values {0.0, 1.0}.

    Any value greater than 0.5 is treated as foreground.
    """
    if mask is None:
        raise ValueError("mask must not be None.")
    return (mask > 0.5).float()


def build_prediction_mask(
    prediction_prob: Optional[Tensor] = None,
    prediction_mask: Optional[Tensor] = None,
    threshold: float = 0.5,
) -> Tensor:
    """
    Build a normalized binary prediction mask from either probability or mask input.

    Exactly one of `prediction_prob` or `prediction_mask` must be provided.
    """
    if prediction_prob is None and prediction_mask is None:
        raise ValueError(
            "build_prediction_mask requires prediction_prob or prediction_mask."
        )
    if prediction_prob is not None and prediction_mask is not None:
        raise ValueError(
            "build_prediction_mask expects exactly one input source."
        )
    if prediction_prob is not None:
        return threshold_probabilities(prediction_prob, threshold)
    return ensure_binary_mask(prediction_mask)


def build_ground_truth_mask(ground_truth_mask: Tensor) -> Tensor:
    """Normalize GT mask to strict binary values {0.0, 1.0}."""
    return ensure_binary_mask(ground_truth_mask)


def _validate_threshold(threshold: float) -> None:
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"Threshold must be in [0, 1], got {threshold}.")

