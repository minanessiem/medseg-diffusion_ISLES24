"""Display-name registry for reporting tables."""

from __future__ import annotations

from typing import Dict


DEFAULT_COLUMN_DISPLAY_NAMES: Dict[str, str] = {
    "roi": "ROI",
    "pos": "Positive Weighting",
    "run_name": "Run Name",
    "run_dir": "Run Directory",
    "status": "Status",
    "best_step": "Best Step",
    "primary_metric_value": "Primary Metric",
    "selected_metrics_csv": "Selected Metrics CSV",
}


DEFAULT_METRIC_DISPLAY_NAMES: Dict[str, str] = {
    "dice_2d_fg": "2D Foreground Dice",
    "dice_3d": "3D Dice",
    "surface_dice_monai_3d": "3D MONAI Surface Dice",
    "hd95_3d": "3D HD95",
    "hd95_medpy_3d": "3D MedPy HD95",
    "abs_volume_diff_3d": "3D Absolute Volume Difference",
    "abs_lesion_count_diff_3d": "3D Absolute Lesion Count Difference",
    "lesion_f1_3d": "3D Lesion F1",
    "empty_volumes": "Empty Volumes",
    "foreground_volume_ratio": "Foreground Volume Ratio",
    "foreground_volumes": "Foreground Volumes",
    "total_volumes": "Total Volumes",
}


def get_display_name(key: str) -> str:
    """Return a publication-friendly display name for a column or metric key."""
    if key in DEFAULT_COLUMN_DISPLAY_NAMES:
        return DEFAULT_COLUMN_DISPLAY_NAMES[key]
    if key in DEFAULT_METRIC_DISPLAY_NAMES:
        return DEFAULT_METRIC_DISPLAY_NAMES[key]
    return key

