"""
2D metrics registry for greenfield evaluation.

Metric keys intentionally match the class names defined in
`src/metrics/metrics.py` (no report-time renaming). Validation aliases are
accepted only as config inputs and resolved to these class-name keys.
"""

from typing import Dict, Iterable, Optional, Sequence, Tuple

from torch import Tensor

from src.metrics.metrics import (
    Dice2DForegroundOnly,
    VoxelF1Score2D,
    VoxelF2Score2D,
    VoxelPrecision2D,
    VoxelSensitivity2D,
    VoxelSpecificity2D,
)

# Class-name keys, as requested.
TWOD_METRIC_CLASSES: Dict[str, type] = {
    "Dice2DForegroundOnly": Dice2DForegroundOnly,
    "VoxelPrecision2D": VoxelPrecision2D,
    "VoxelSensitivity2D": VoxelSensitivity2D,
    "VoxelSpecificity2D": VoxelSpecificity2D,
    "VoxelF1Score2D": VoxelF1Score2D,
    "VoxelF2Score2D": VoxelF2Score2D,
}

TWOD_VALIDATION_METRIC_ALIASES: Dict[str, Tuple[str, ...]] = {
    "dice": ("Dice2DForegroundOnly",),
    "dice_2d": ("Dice2DForegroundOnly",),
    "dice_2d_fg": ("Dice2DForegroundOnly",),
    "precision": ("VoxelPrecision2D",),
    "precision_2d": ("VoxelPrecision2D",),
    "recall": ("VoxelSensitivity2D",),
    "recall_2d": ("VoxelSensitivity2D",),
    "sensitivity": ("VoxelSensitivity2D",),
    "sensitivity_2d": ("VoxelSensitivity2D",),
    "specificity": ("VoxelSpecificity2D",),
    "specificity_2d": ("VoxelSpecificity2D",),
    "f1": ("VoxelF1Score2D",),
    "f1_2d": ("VoxelF1Score2D",),
    "f2": ("VoxelF2Score2D",),
    "f2_2d": ("VoxelF2Score2D",),
}

# Backward-compatible public name used by tests and older evaluation imports.
METRIC_REGISTRY: Dict[str, type] = {}


def resolve_2d_metric_class_names(metric_names: Iterable[str]) -> Tuple[str, ...]:
    """
    Resolve validation metric aliases or class-name keys to class-name keys.
    """
    resolved = []
    seen = set()
    for raw_name in metric_names:
        name = str(raw_name).strip()
        if not name:
            continue
        if name in TWOD_METRIC_CLASSES:
            class_names = (name,)
        elif name in TWOD_VALIDATION_METRIC_ALIASES:
            class_names = TWOD_VALIDATION_METRIC_ALIASES[name]
        else:
            available = sorted(set(TWOD_METRIC_CLASSES) | set(TWOD_VALIDATION_METRIC_ALIASES))
            raise ValueError(
                f"Unknown 2D metric name '{name}'. Expected a metric class name "
                f"or known validation alias. Available: {available}"
            )
        for class_name in class_names:
            if class_name not in seen:
                resolved.append(class_name)
                seen.add(class_name)
    if not resolved:
        raise ValueError("2D metric selection cannot be empty.")
    return tuple(resolved)


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


class DiceAtThreshold(ThresholdMetricBase):
    """Dice coefficient at a given threshold."""

    metric_class = Dice2DForegroundOnly


class PrecisionAtThreshold(ThresholdMetricBase):
    """Precision at a given threshold."""

    metric_class = VoxelPrecision2D


class RecallAtThreshold(ThresholdMetricBase):
    """Recall/sensitivity at a given threshold."""

    metric_class = VoxelSensitivity2D


class SpecificityAtThreshold(ThresholdMetricBase):
    """Specificity at a given threshold."""

    metric_class = VoxelSpecificity2D


class F1AtThreshold(ThresholdMetricBase):
    """F1 score at a given threshold."""

    metric_class = VoxelF1Score2D


class F2AtThreshold(ThresholdMetricBase):
    """F2 score at a given threshold."""

    metric_class = VoxelF2Score2D


TWOD_METRIC_WRAPPERS: Dict[str, type] = {
    "Dice2DForegroundOnly": DiceAtThreshold,
    "VoxelPrecision2D": PrecisionAtThreshold,
    "VoxelSensitivity2D": RecallAtThreshold,
    "VoxelSpecificity2D": SpecificityAtThreshold,
    "VoxelF1Score2D": F1AtThreshold,
    "VoxelF2Score2D": F2AtThreshold,
}
METRIC_REGISTRY = TWOD_METRIC_WRAPPERS


def get_all_metrics(
    metric_names: Optional[Iterable[str]] = None,
) -> Dict[str, ThresholdMetricBase]:
    """Return instantiated wrappers for all registered metrics."""
    selected_names: Sequence[str]
    if metric_names is None:
        selected_names = tuple(TWOD_METRIC_CLASSES.keys())
    else:
        selected_names = resolve_2d_metric_class_names(metric_names)
    return {name: TWOD_METRIC_WRAPPERS[name]() for name in selected_names}


def compute_metrics_at_threshold(
    pred: Tensor,
    gt: Tensor,
    threshold: float,
    metric_names: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """
    Compute all registered metrics at a given threshold.

    Args:
        pred: prediction tensor
        gt: ground truth binary mask tensor
        threshold: threshold used for prediction binarization

    Returns:
        Mapping of metric name to scalar metric value.
    """
    metrics = get_all_metrics(metric_names=metric_names)
    return {name: metric(pred, gt, threshold) for name, metric in metrics.items()}

