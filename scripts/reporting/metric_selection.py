"""Best-checkpoint metric selection for reporting."""

from __future__ import annotations

import math
from typing import Iterable

from scripts.reporting.schema import MetricSnapshot


VALID_DIRECTIONS = {"max", "min"}


def validate_direction(direction: str) -> str:
    """Validate and normalize a metric direction."""
    normalized = str(direction).strip().lower()
    if normalized not in VALID_DIRECTIONS:
        allowed = ", ".join(sorted(VALID_DIRECTIONS))
        raise ValueError(f"Unsupported metric direction '{direction}'. Allowed: {allowed}.")
    return normalized


def is_better(value: float, best_value: float, direction: str) -> bool:
    """Return whether ``value`` improves over ``best_value``."""
    direction = validate_direction(direction)
    if direction == "max":
        return value > best_value
    return value < best_value


def select_best_snapshot(
    snapshots: Iterable[MetricSnapshot],
    primary_metric: str,
    primary_direction: str,
) -> MetricSnapshot:
    """
    Select one metrics CSV snapshot by primary metric and direction.

    Selection is based on CSV contents, not filename ordering.
    """
    direction = validate_direction(primary_direction)
    best_snapshot: MetricSnapshot | None = None
    best_value: float | None = None
    missing_metric_count = 0

    for snapshot in snapshots:
        if primary_metric not in snapshot.metrics:
            missing_metric_count += 1
            continue
        value = float(snapshot.metrics[primary_metric])
        if math.isnan(value):
            continue
        if best_value is None or is_better(value, best_value, direction):
            best_value = value
            best_snapshot = snapshot

    if best_snapshot is None:
        if missing_metric_count:
            raise KeyError(
                f"Primary metric '{primary_metric}' was not found in any candidate "
                "best-checkpoint metrics CSV."
            )
        raise ValueError(
            f"No finite values for primary metric '{primary_metric}' were found."
        )

    return best_snapshot

