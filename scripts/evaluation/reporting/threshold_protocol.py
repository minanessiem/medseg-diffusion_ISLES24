"""
Threshold protocol helpers for greenfield evaluation.

This module encapsulates threshold policy logic and keeps orchestration
code thin in entrypoint scripts.
"""

from dataclasses import replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf

from scripts.evaluation.core.contracts import (
    EvaluationThresholdProtocol,
    PrimaryMetricSelector,
    ThresholdProtocol,
)
from scripts.evaluation.metrics.registry_2d import resolve_2d_metric_class_names

DEFAULT_METRIC_NAMES: Tuple[str, ...] = (
    "dice",
    "precision",
    "recall",
    "specificity",
    "f1",
    "f2",
)
EVALUATION_THRESHOLD_MODES: Tuple[str, ...] = (
    "fixed",
    "sweep",
    "oracle_per_case",
    "sweep_with_oracle",
)
PRIMARY_LEVELS: Tuple[str, ...] = ("slice", "volume")
LEVEL_ALIASES = {
    "slice": "slice",
    "slices": "slice",
    "volume": "volume",
    "volumes": "volume",
}
PRIMARY_STATISTICS: Tuple[str, ...] = ("mean", "median")
PRIMARY_DIRECTIONS: Tuple[str, ...] = ("max", "min")


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


def build_evaluation_threshold_protocol(
    cfg: DictConfig,
) -> EvaluationThresholdProtocol:
    """
    Build the extended evaluation threshold protocol from config.
    """
    protocol_path = "evaluation.threshold_protocol"
    if OmegaConf.select(cfg, protocol_path, default=None) is None:
        raise ValueError("Missing required config section: evaluation.threshold_protocol.")

    mode = str(
        OmegaConf.select(cfg, f"{protocol_path}.mode", default="fixed")
    ).strip()
    if mode not in EVALUATION_THRESHOLD_MODES:
        allowed = ", ".join(EVALUATION_THRESHOLD_MODES)
        raise ValueError(f"Unsupported threshold protocol mode '{mode}'. Allowed: {allowed}.")

    fixed_threshold = float(
        OmegaConf.select(cfg, f"{protocol_path}.fixed_threshold", default=0.5)
    )
    _validate_threshold(fixed_threshold)

    if mode == "fixed":
        thresholds = [fixed_threshold]
    else:
        threshold_spec = OmegaConf.select(
            cfg,
            f"{protocol_path}.thresholds",
            default=str(fixed_threshold),
        )
        thresholds = _parse_threshold_values(threshold_spec)
        thresholds = _include_threshold(thresholds, fixed_threshold)

    primary = build_primary_metric_selector(cfg)
    return EvaluationThresholdProtocol(
        mode=mode,  # type: ignore[arg-type]
        thresholds=thresholds,
        fixed_threshold=fixed_threshold,
        primary=primary,
    )


def build_primary_metric_selector(cfg: DictConfig) -> PrimaryMetricSelector:
    """
    Build and validate the primary metric selector from config.
    """
    primary_path = "evaluation.threshold_protocol.primary"
    level = normalize_evaluation_level(
        OmegaConf.select(cfg, f"{primary_path}.level", default="volume")
    )
    metric = str(
        OmegaConf.select(
            cfg,
            f"{primary_path}.metric",
            default="DiceNativeCoefficient",
        )
    ).strip()
    statistic = str(
        OmegaConf.select(cfg, f"{primary_path}.statistic", default="mean")
    ).strip()
    direction = str(
        OmegaConf.select(cfg, f"{primary_path}.direction", default="max")
    ).strip()

    if not metric:
        raise ValueError("Primary metric name must not be empty.")
    if statistic not in PRIMARY_STATISTICS:
        allowed = ", ".join(PRIMARY_STATISTICS)
        raise ValueError(
            f"Unsupported primary metric statistic '{statistic}'. Allowed: {allowed}."
        )
    if direction not in PRIMARY_DIRECTIONS:
        allowed = ", ".join(PRIMARY_DIRECTIONS)
        raise ValueError(
            f"Unsupported primary metric direction '{direction}'. Allowed: {allowed}."
        )

    return PrimaryMetricSelector(
        level=level,  # type: ignore[arg-type]
        metric=metric,
        statistic=statistic,  # type: ignore[arg-type]
        direction=direction,  # type: ignore[arg-type]
    )


def normalize_evaluation_level(value: object) -> str:
    """
    Normalize user-facing evaluation level tokens to internal canonical values.
    """
    token = str(value).strip().lower()
    if token in LEVEL_ALIASES:
        return LEVEL_ALIASES[token]
    allowed = ", ".join(sorted(LEVEL_ALIASES))
    raise ValueError(f"Unsupported evaluation level '{value}'. Allowed: {allowed}.")


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
    result_metrics = finalized_results[float(protocol.thresholds[0])]["metrics"]
    if metric_name not in result_metrics:
        resolved_names = resolve_2d_metric_class_names([metric_name])
        if len(resolved_names) == 1 and resolved_names[0] in result_metrics:
            metric_name = resolved_names[0]
    if selection_scope is None:
        selection_scope = (
            "foreground_only"
            if metric_name in {"dice", "Dice2DForegroundOnly"}
            else "all_slices"
        )

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


def _parse_threshold_values(threshold_spec: Any) -> List[float]:
    if isinstance(threshold_spec, str):
        thresholds = parse_thresholds(threshold_spec)
    elif isinstance(threshold_spec, (list, tuple, ListConfig)) or OmegaConf.is_list(threshold_spec):
        thresholds = [float(value) for value in list(threshold_spec)]
    elif isinstance(threshold_spec, (int, float)):
        thresholds = [float(threshold_spec)]
    else:
        raise ValueError(
            "Thresholds must be a string range/list, numeric value, or sequence. "
            f"Got {type(threshold_spec).__name__}."
        )
    return _normalize_thresholds(thresholds)


def _include_threshold(thresholds: Sequence[float], threshold: float) -> List[float]:
    return _normalize_thresholds([*thresholds, float(threshold)])


def _normalize_thresholds(thresholds: Sequence[float]) -> List[float]:
    normalized: List[float] = []
    seen = set()
    for threshold in thresholds:
        value = round(float(threshold), 4)
        _validate_threshold(value)
        if value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    if not normalized:
        raise ValueError("Threshold protocol requires at least one threshold.")
    return sorted(normalized)

