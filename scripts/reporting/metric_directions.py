"""Metric direction registry for sorting and highlighting."""

from __future__ import annotations

from typing import Dict, Iterable

from scripts.reporting.metric_selection import validate_direction


DEFAULT_METRIC_DIRECTIONS: Dict[str, str] = {
    "dice_2d_fg": "max",
    "dice_3d": "max",
    "surface_dice_monai_3d": "max",
    "hd95_3d": "min",
    "hd95_medpy_3d": "min",
    "abs_volume_diff_3d": "min",
    "abs_lesion_count_diff_3d": "min",
    "lesion_f1_3d": "max",
}


def parse_metric_direction(spec: str) -> tuple[str, str]:
    """Parse ``metric=max|min`` CLI syntax."""
    if "=" not in spec:
        raise ValueError(
            f"Invalid --metric-direction value '{spec}'. Expected metric=max|min."
        )
    metric, direction = spec.split("=", 1)
    metric = metric.strip()
    if not metric:
        raise ValueError(f"Invalid --metric-direction value '{spec}': metric is empty.")
    return metric, validate_direction(direction)


def build_metric_directions(
    specs: Iterable[str],
    primary_metric: str,
    primary_direction: str,
) -> Dict[str, str]:
    """Build metric directions using defaults, primary fallback, then overrides."""
    directions = dict(DEFAULT_METRIC_DIRECTIONS)
    directions[primary_metric] = validate_direction(primary_direction)
    for spec in specs:
        metric, direction = parse_metric_direction(spec)
        directions[metric] = direction
    return directions


def get_metric_direction(
    metric_name: str,
    directions: Dict[str, str],
    default: str = "max",
) -> str:
    """Return a validated direction for a metric."""
    return validate_direction(directions.get(metric_name, default))

