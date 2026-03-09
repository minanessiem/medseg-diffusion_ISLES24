"""
3D metrics registry for greenfield evaluation.

Metric keys intentionally match the class names defined in
`src/metrics/metrics.py` (no renaming).
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, MutableMapping, Optional

from torch import Tensor

from src.metrics.metrics import (
    AbsoluteLesionCountDifferenceCC3D,
    AbsoluteVolumeDifferenceNative,
    DiceMedpyCoefficient,
    DiceNativeCoefficient,
    GroundTruthVolumeMm3,
    HausdorffDistance95Medpy,
    HausdorffDistance95MonaiMm,
    HausdorffDistance95Native,
    LesionF1CC3DScore,
    PredictedVolumeMm3,
    SurfaceDiceMonai,
    VoxelFalseNegatives,
    VoxelFalsePositives,
    VoxelTrueNegatives,
    VoxelTruePositives,
)

# Class-name keys, as requested.
THREED_METRIC_DEFAULT_CONFIGS: Dict[str, Dict[str, object]] = {
    "DiceMedpyCoefficient": {},
    "DiceNativeCoefficient": {},
    "AbsoluteVolumeDifferenceNative": {"voxel_size": 1.0},
    "AbsoluteLesionCountDifferenceCC3D": {},
    "LesionF1CC3DScore": {},
    "HausdorffDistance95Medpy": {},
    "HausdorffDistance95Native": {},
    "HausdorffDistance95MonaiMm": {"spacing": (1.0, 1.0, 1.0)},
    "SurfaceDiceMonai": {"spacing": (1.0, 1.0, 1.0), "tolerance_mm": 1.0},
    "VoxelTruePositives": {},
    "VoxelFalsePositives": {},
    "VoxelFalseNegatives": {},
    "VoxelTrueNegatives": {},
    "PredictedVolumeMm3": {"spacing": (1.0, 1.0, 1.0)},
    "GroundTruthVolumeMm3": {"spacing": (1.0, 1.0, 1.0)},
}

THREED_METRIC_CLASSES: Dict[str, type] = {
    "DiceMedpyCoefficient": DiceMedpyCoefficient,
    "DiceNativeCoefficient": DiceNativeCoefficient,
    "AbsoluteVolumeDifferenceNative": AbsoluteVolumeDifferenceNative,
    "AbsoluteLesionCountDifferenceCC3D": AbsoluteLesionCountDifferenceCC3D,
    "LesionF1CC3DScore": LesionF1CC3DScore,
    "HausdorffDistance95Medpy": HausdorffDistance95Medpy,
    "HausdorffDistance95Native": HausdorffDistance95Native,
    "HausdorffDistance95MonaiMm": HausdorffDistance95MonaiMm,
    "SurfaceDiceMonai": SurfaceDiceMonai,
    "VoxelTruePositives": VoxelTruePositives,
    "VoxelFalsePositives": VoxelFalsePositives,
    "VoxelFalseNegatives": VoxelFalseNegatives,
    "VoxelTrueNegatives": VoxelTrueNegatives,
    "PredictedVolumeMm3": PredictedVolumeMm3,
    "GroundTruthVolumeMm3": GroundTruthVolumeMm3,
}


class ThresholdMetric3DBase:
    """Threshold-aware wrapper for a 3D metric implementation."""

    metric_name: str
    metric_class: type

    def __init__(self, metric_name: str, metric_class: type, metric_kwargs: Mapping[str, object]):
        self.metric_name = metric_name
        self.metric_class = metric_class
        self.metric_kwargs = dict(metric_kwargs)
        self._metric = self.metric_class(**self.metric_kwargs)

    def __call__(self, pred: Tensor, gt: Tensor, threshold: float = 0.5) -> float:
        self._reset_metric()
        pred_volume = _ensure_batched_channel_first_volume(pred)
        gt_volume = _ensure_batched_channel_first_volume(gt)
        pred_binary = (pred_volume > threshold).float()
        result = self._metric(pred_binary, gt_volume)
        if isinstance(result, Tensor):
            return float(result.item())
        return float(result)

    def _reset_metric(self) -> None:
        """
        Reset current metric instance.

        Most metric classes support reset/compute lifecycle. If reset fails for
        any legacy implementation, recreate the metric to preserve correctness.
        """
        try:
            self._metric.reset()
        except Exception:
            self._metric = self.metric_class(**self.metric_kwargs)


def get_all_metrics_3d(
    metric_configs: Optional[Mapping[str, Mapping[str, object]]] = None,
    metric_names: Optional[Iterable[str]] = None,
) -> Dict[str, ThresholdMetric3DBase]:
    """
    Return instantiated 3D metric wrappers.

    Args:
        metric_configs: Optional override map keyed by metric class name.
    """
    configs: MutableMapping[str, Mapping[str, object]] = dict(THREED_METRIC_DEFAULT_CONFIGS)
    if metric_configs is not None:
        for name, cfg in metric_configs.items():
            if name not in THREED_METRIC_CLASSES:
                raise ValueError(f"Unknown 3D metric class name: {name}")
            configs[name] = dict(cfg)
    selected_names = list(metric_names) if metric_names is not None else list(THREED_METRIC_CLASSES.keys())
    for name in selected_names:
        if name not in THREED_METRIC_CLASSES:
            raise ValueError(f"Unknown 3D metric class name: {name}")
    return {
        name: ThresholdMetric3DBase(
            metric_name=name,
            metric_class=THREED_METRIC_CLASSES[name],
            metric_kwargs=configs.get(name, {}),
        )
        for name in selected_names
    }


def compute_metrics_3d_at_threshold(
    pred: Tensor,
    gt: Tensor,
    threshold: float,
    metric_configs: Optional[Mapping[str, Mapping[str, object]]] = None,
    metric_names: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """Compute all registered 3D metrics at one threshold."""
    metrics = get_all_metrics_3d(metric_configs=metric_configs, metric_names=metric_names)
    return {name: metric(pred, gt, threshold) for name, metric in metrics.items()}


def _ensure_batched_channel_first_volume(volume: Tensor) -> Tensor:
    """
    Normalize volume tensor to shape [B, C, H, W, D].
    """
    if volume.ndim == 5:
        return volume
    if volume.ndim == 4:
        return volume.unsqueeze(0)
    if volume.ndim == 3:
        return volume.unsqueeze(0).unsqueeze(0)
    raise ValueError(
        f"Expected volume tensor with 3, 4, or 5 dims. Got shape={tuple(volume.shape)}."
    )
