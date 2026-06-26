"""Build ROI x parameter style pivot tables."""

from __future__ import annotations

import math
from typing import Any, Dict, Sequence

from scripts.reporting.metric_directions import get_metric_direction
from scripts.reporting.metric_selection import is_better
from scripts.reporting.schema import GridTable, ParamAlias, RunSummary


def build_grid_tables(
    summaries: Sequence[RunSummary],
    row_alias: str,
    col_alias: str,
    metric_names: Sequence[str],
    param_aliases: Sequence[ParamAlias],
    metric_directions: Dict[str, str],
) -> list[GridTable]:
    """Build one pivot table per requested metric."""
    alias_names = {alias.alias for alias in param_aliases}
    if row_alias not in alias_names:
        raise ValueError(f"Grid row alias '{row_alias}' was not defined with --param.")
    if col_alias not in alias_names:
        raise ValueError(f"Grid col alias '{col_alias}' was not defined with --param.")
    if not metric_names:
        raise ValueError("At least one --grid-values metric is required for grid output.")

    complete = [summary for summary in summaries if summary.status == "complete"]
    row_values = _sorted_unique(summary.params[row_alias] for summary in complete)
    col_values = _sorted_unique(summary.params[col_alias] for summary in complete)

    tables: list[GridTable] = []
    for metric_name in metric_names:
        cells: Dict[tuple[Any, Any], Any] = {}
        owners: Dict[tuple[Any, Any], str] = {}
        for summary in complete:
            row_value = summary.params[row_alias]
            col_value = summary.params[col_alias]
            key = (row_value, col_value)
            if key in cells:
                raise ValueError(
                    "Duplicate grid cell for "
                    f"{row_alias}={row_value}, {col_alias}={col_value}: "
                    f"{owners[key]} and {summary.run_name}."
                )
            cells[key] = summary.metrics.get(metric_name)
            owners[key] = summary.run_name
        tables.append(
            GridTable(
                metric_name=metric_name,
                row_alias=row_alias,
                col_alias=col_alias,
                row_values=row_values,
                col_values=col_values,
                cells=cells,
                highlighted_cells=_highlight_best_cells(
                    cells=cells,
                    metric_name=metric_name,
                    metric_directions=metric_directions,
                ),
            )
        )
    return tables


def _sorted_unique(values: Sequence[Any]) -> list[Any]:
    return sorted(set(values), key=_sort_value)


def _sort_value(value: Any) -> tuple[str, Any]:
    if isinstance(value, (int, float)):
        return ("0", value)
    return ("1", str(value))


def _highlight_best_cells(
    cells: Dict[tuple[Any, Any], Any],
    metric_name: str,
    metric_directions: Dict[str, str],
) -> set[tuple[Any, Any]]:
    numeric_values: list[tuple[tuple[Any, Any], float]] = []
    for key, value in cells.items():
        if isinstance(value, (int, float)):
            numeric = float(value)
            if not math.isnan(numeric):
                numeric_values.append((key, numeric))
    if not numeric_values:
        return set()

    direction = get_metric_direction(metric_name, metric_directions)
    best_key, best_value = numeric_values[0]
    best_keys = {best_key}
    for key, value in numeric_values[1:]:
        if is_better(value, best_value, direction):
            best_value = value
            best_keys = {key}
        elif value == best_value:
            best_keys.add(key)
    return best_keys

