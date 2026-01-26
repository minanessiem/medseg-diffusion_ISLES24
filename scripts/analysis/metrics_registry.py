"""
Metrics registry for threshold analysis.

All implementations reference src/metrics/metrics.py to ensure
consistency with training/validation metrics.

The wrappers apply a configurable threshold before passing predictions
to the underlying metric classes, which internally use 0.5 as their
binarization threshold.
"""

from typing import Dict, Callable, Any
import torch
from torch import Tensor

# Import existing 2D metrics from src/metrics/metrics.py
from src.metrics.metrics import (
    Dice2DForegroundOnly,
    VoxelPrecision2D,
    VoxelSensitivity2D,
    VoxelSpecificity2D,
    VoxelF1Score2D,
    VoxelF2Score2D,
)

# Registry dictionary mapping metric names to wrapper classes
METRIC_REGISTRY: Dict[str, type] = {}


def register_metric(name: str):
    """Decorator to register a metric wrapper class."""
    def decorator(cls):
        METRIC_REGISTRY[name] = cls
        return cls
    return decorator


class ThresholdMetricBase:
    """
    Base class for threshold-aware metric wrappers.
    
    Subclasses must define `metric_class` pointing to the underlying
    metric from src/metrics/metrics.py.
    """
    metric_class: type = None
    
    def __init__(self):
        if self.metric_class is None:
            raise NotImplementedError("Subclass must define metric_class")
        self._metric = self.metric_class()
    
    def __call__(self, pred: Tensor, gt: Tensor, threshold: float = 0.5) -> float:
        """
        Compute metric at the given threshold.
        
        Args:
            pred: Raw sigmoid predictions (values in [0, 1])
            gt: Ground truth binary mask
            threshold: Binarization threshold for predictions
            
        Returns:
            Metric value as a Python float
        """
        # Reset the metric to ensure fresh computation
        self._metric.reset()
        
        # Binarize predictions at the given threshold
        # Convert to float (0.0 or 1.0) so the underlying metric's
        # internal 0.5 threshold works correctly
        pred_binary = (pred > threshold).float()
        
        # Compute the metric
        result = self._metric(pred_binary, gt)
        
        # Return as Python float
        if isinstance(result, Tensor):
            return result.item()
        return float(result)


@register_metric('dice')
class DiceAtThreshold(ThresholdMetricBase):
    """
    Dice coefficient at a given threshold.
    Wraps Dice2DForegroundOnly from src/metrics/metrics.py.
    """
    metric_class = Dice2DForegroundOnly


@register_metric('precision')
class PrecisionAtThreshold(ThresholdMetricBase):
    """
    Precision (positive predictive value) at a given threshold.
    Wraps VoxelPrecision2D from src/metrics/metrics.py.
    """
    metric_class = VoxelPrecision2D


@register_metric('recall')
class RecallAtThreshold(ThresholdMetricBase):
    """
    Recall (sensitivity/true positive rate) at a given threshold.
    Wraps VoxelSensitivity2D from src/metrics/metrics.py.
    """
    metric_class = VoxelSensitivity2D


@register_metric('specificity')
class SpecificityAtThreshold(ThresholdMetricBase):
    """
    Specificity (true negative rate) at a given threshold.
    Wraps VoxelSpecificity2D from src/metrics/metrics.py.
    """
    metric_class = VoxelSpecificity2D


@register_metric('f1')
class F1AtThreshold(ThresholdMetricBase):
    """
    F1 score at a given threshold.
    Wraps VoxelF1Score2D from src/metrics/metrics.py.
    """
    metric_class = VoxelF1Score2D


@register_metric('f2')
class F2AtThreshold(ThresholdMetricBase):
    """
    F2 score (recall-weighted) at a given threshold.
    Wraps VoxelF2Score2D from src/metrics/metrics.py.
    """
    metric_class = VoxelF2Score2D


def get_all_metrics() -> Dict[str, ThresholdMetricBase]:
    """
    Returns instantiated metric wrappers for all registered metrics.
    
    Returns:
        Dictionary mapping metric names to instantiated wrapper objects.
    """
    return {name: cls() for name, cls in METRIC_REGISTRY.items()}


def compute_metrics_at_threshold(
    pred: Tensor,
    gt: Tensor,
    threshold: float
) -> Dict[str, float]:
    """
    Compute all registered metrics at a given threshold.
    
    Args:
        pred: Raw sigmoid predictions (values in [0, 1])
        gt: Ground truth binary mask
        threshold: Binarization threshold for predictions
        
    Returns:
        Dictionary mapping metric names to computed values.
    """
    metrics = get_all_metrics()
    results = {}
    
    for name, metric in metrics.items():
        results[name] = metric(pred, gt, threshold)
    
    return results

