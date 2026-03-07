"""
Threshold protocol helpers for greenfield evaluation.

This module encapsulates threshold policy logic and keeps orchestration
code thin in entrypoint scripts.
"""

from dataclasses import replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from scripts.evaluation.contracts import ThresholdProtocol

DEFAULT_METRIC_NAMES: Tuple[str, ...] = (
    "dice",
    "precision",
    "recall",
    "specificity",
    "f1",
    "f2",
)


def parse_thresholds(threshold_spec: str) -> List[float]:
    """
    Parse thresholds from range or csv notation.

    Supported formats:
    - "start:stop:step" (inclusive stop with numeric tolerance)
    - "0.1,0.3,0.5"
    """
    spec = threshold_spec.strip()
    if not spec:
        raise ValueError("Threshold specification must not be empty.")

    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid threshold range '{threshold_spec}'. Expected start:stop:step."
            )
        start, stop, step = (float(parts[0]), float(parts[1]), float(parts[2]))
        if step <= 0:
            raise ValueError(f"Threshold step must be > 0, got {step}.")
        thresholds = np.arange(start, stop + step / 2.0, step)
        return [round(float(t), 4) for t in thresholds]

    parsed = [float(token.strip()) for token in spec.split(",") if token.strip()]
    if not parsed:
        raise ValueError("Threshold specification did not contain any values.")
    return parsed


def make_fixed_protocol(threshold: float) -> ThresholdProtocol:
    """Create a fixed-threshold protocol."""
    _validate_threshold(threshold)
    return ThresholdProtocol(mode="fixed", thresholds=[float(threshold)], optimize_metric=None)


def make_sweep_protocol(
    thresholds: Sequence[float],
    optimize_metric: str = "dice",
) -> ThresholdProtocol:
    """Create a sweep protocol with basic validation."""
    normalized = [float(t) for t in thresholds]
    if not normalized:
        raise ValueError("Sweep protocol requires at least one threshold.")
    for threshold in normalized:
        _validate_threshold(threshold)
    if optimize_metric not in DEFAULT_METRIC_NAMES:
        available = ", ".join(DEFAULT_METRIC_NAMES)
        raise ValueError(
            f"Unsupported optimize_metric '{optimize_metric}'. Available: {available}"
        )
    return ThresholdProtocol(
        mode="sweep",
        thresholds=normalized,
        optimize_metric=optimize_metric,
    )


def make_sweep_protocol_from_spec(
    threshold_spec: str,
    optimize_metric: str = "dice",
) -> ThresholdProtocol:
    """Convenience builder for sweep protocol from CLI threshold string."""
    return make_sweep_protocol(parse_thresholds(threshold_spec), optimize_metric=optimize_metric)


def enforce_post_threshold_mode(protocol: ThresholdProtocol) -> ThresholdProtocol:
    """
    Enforce post-threshold behavior for mask-only sources (e.g., nnU-Net).

    If a sweep is provided, this raises to avoid accidental misuse.
    """
    if protocol.mode == "sweep":
        raise ValueError(
            "Post-threshold mask sources do not support threshold sweep protocols."
        )
    if len(protocol.thresholds) != 1:
        raise ValueError(
            "Post-threshold evaluation requires exactly one fixed threshold value."
        )
    return replace(protocol, mode="fixed", optimize_metric=None)


def select_primary_threshold(
    finalized_results: Dict[float, Dict[str, object]],
    protocol: ThresholdProtocol,
    selection_scope: Optional[str] = None,
) -> float:
    """
    Select the primary threshold to report.

    For fixed protocol: returns the only threshold.
    For sweep protocol: selects threshold maximizing optimize_metric mean.
    """
    if protocol.mode == "fixed":
        return float(protocol.thresholds[0])

    metric_name = protocol.optimize_metric or "dice"
    if selection_scope is None:
        selection_scope = "foreground_only" if metric_name == "dice" else "all_slices"

    best_threshold = None
    best_value = None
    for threshold in protocol.thresholds:
        metric_block = finalized_results[float(threshold)]["metrics"][metric_name]
        candidate = float(metric_block[selection_scope]["mean"])
        if best_value is None or candidate > best_value:
            best_value = candidate
            best_threshold = float(threshold)

    if best_threshold is None:
        raise RuntimeError("Unable to select primary threshold from results.")
    return best_threshold


def iter_protocol_thresholds(protocol: ThresholdProtocol) -> Iterable[float]:
    """Yield thresholds in protocol order."""
    return (float(t) for t in protocol.thresholds)


def _validate_threshold(threshold: float) -> None:
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"Threshold must be in [0, 1], got {threshold}.")

