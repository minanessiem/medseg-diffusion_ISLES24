"""
Metrics registry for greenfield evaluation.

All implementations reference src/metrics/metrics.py to ensure consistency
with training/validation metrics and existing evaluation behavior.

The wrappers apply a configurable threshold before passing predictions to
the underlying metric classes, which internally use 0.5 as their own
binarization threshold.
"""

from typing import Dict

from torch import Tensor

from src.metrics.metrics import (
    Dice2DForegroundOnly,
    VoxelF1Score2D,
    VoxelF2Score2D,
    VoxelPrecision2D,
    VoxelSensitivity2D,
    VoxelSpecificity2D,
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

    Subclasses must define `metric_class` pointing to the underlying metric
    implementation from src/metrics/metrics.py.
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
            pred: prediction tensor (probabilities or mask)
            gt: ground truth binary mask tensor
            threshold: binarization threshold for predictions

        Returns:
            Metric value as a Python float.
        """
        self._metric.reset()
        pred_binary = (pred > threshold).float()
        result = self._metric(pred_binary, gt)
        if isinstance(result, Tensor):
            return float(result.item())
        return float(result)


@register_metric("dice")
class DiceAtThreshold(ThresholdMetricBase):
    """Dice coefficient at a given threshold."""

    metric_class = Dice2DForegroundOnly


@register_metric("precision")
class PrecisionAtThreshold(ThresholdMetricBase):
    """Precision at a given threshold."""

    metric_class = VoxelPrecision2D


@register_metric("recall")
class RecallAtThreshold(ThresholdMetricBase):
    """Recall/sensitivity at a given threshold."""

    metric_class = VoxelSensitivity2D


@register_metric("specificity")
class SpecificityAtThreshold(ThresholdMetricBase):
    """Specificity at a given threshold."""

    metric_class = VoxelSpecificity2D


@register_metric("f1")
class F1AtThreshold(ThresholdMetricBase):
    """F1 score at a given threshold."""

    metric_class = VoxelF1Score2D


@register_metric("f2")
class F2AtThreshold(ThresholdMetricBase):
    """F2 score at a given threshold."""

    metric_class = VoxelF2Score2D


def get_all_metrics() -> Dict[str, ThresholdMetricBase]:
    """Return instantiated wrappers for all registered metrics."""
    return {name: cls() for name, cls in METRIC_REGISTRY.items()}


def compute_metrics_at_threshold(pred: Tensor, gt: Tensor, threshold: float) -> Dict[str, float]:
    """
    Compute all registered metrics at a given threshold.

    Args:
        pred: prediction tensor
        gt: ground truth binary mask tensor
        threshold: threshold used for prediction binarization

    Returns:
        Mapping of metric name to scalar metric value.
    """
    metrics = get_all_metrics()
    return {name: metric(pred, gt, threshold) for name, metric in metrics.items()}

