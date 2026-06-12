"""
Generic threshold selection helpers for evaluation aggregate rows.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Optional

from scripts.evaluation.core.contracts import PrimaryMetricSelector


def select_best_threshold_from_rows(
    rows: Sequence[Mapping[str, Any]],
    selector: PrimaryMetricSelector,
) -> float:
    """
    Select the best threshold from aggregate rows using ``selector``.
    """
    if not rows:
        raise ValueError("Cannot select threshold from an empty row collection.")

    best_threshold: Optional[float] = None
    best_value: Optional[float] = None
    for row in rows:
        if not _row_matches_level(row, selector.level):
            continue

        threshold = _extract_threshold(row)
        value = _extract_metric_statistic(row, selector)
        if math.isnan(value):
            continue

        if best_value is None or _is_better(value, best_value, selector.direction):
            best_value = value
            best_threshold = threshold

    if best_threshold is None:
        raise ValueError(
            "Unable to select threshold. No rows matched selector "
            f"level={selector.level!r}, metric={selector.metric!r}, "
            f"statistic={selector.statistic!r}."
        )
    return float(best_threshold)


def _row_matches_level(row: Mapping[str, Any], level: str) -> bool:
    row_level = row.get("level")
    if row_level is None:
        return True
    return str(row_level) == level


def _extract_threshold(row: Mapping[str, Any]) -> float:
    if "threshold" not in row:
        raise ValueError(f"Threshold row is missing required 'threshold' key: {row}")
    return float(row["threshold"])


def _extract_metric_statistic(
    row: Mapping[str, Any],
    selector: PrimaryMetricSelector,
) -> float:
    direct_key = f"{selector.metric}_{selector.statistic}"
    if direct_key in row:
        return float(row[direct_key])

    metrics = row.get("metrics")
    if isinstance(metrics, Mapping) and selector.metric in metrics:
        return _extract_statistic_from_metric_block(
            metric_block=metrics[selector.metric],
            selector=selector,
        )

    if selector.metric in row:
        return _extract_statistic_from_metric_block(
            metric_block=row[selector.metric],
            selector=selector,
        )

    raise ValueError(
        f"Metric '{selector.metric}' not found in threshold row at "
        f"threshold={row.get('threshold')}."
    )


def _extract_statistic_from_metric_block(
    metric_block: Any,
    selector: PrimaryMetricSelector,
) -> float:
    if isinstance(metric_block, Mapping):
        if selector.statistic in metric_block:
            return float(metric_block[selector.statistic])

        # Compatibility with current slice-level scoped report blocks.
        for scope in ("all_slices", "foreground_only"):
            scope_block = metric_block.get(scope)
            if isinstance(scope_block, Mapping) and selector.statistic in scope_block:
                return float(scope_block[selector.statistic])

    if selector.statistic == "mean" and isinstance(metric_block, (int, float)):
        return float(metric_block)

    raise ValueError(
        f"Statistic '{selector.statistic}' not found for metric '{selector.metric}'."
    )


def _is_better(value: float, best_value: float, direction: str) -> bool:
    if direction == "max":
        return value > best_value
    if direction == "min":
        return value < best_value
    raise ValueError(f"Unsupported selector direction '{direction}'.")
