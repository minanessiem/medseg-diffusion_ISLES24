"""Build flat experiment summary tables."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Sequence

from scripts.reporting.metric_directions import get_metric_direction
from scripts.reporting.metric_selection import is_better
from scripts.reporting.schema import FlatTable, ParamAlias, RunSummary


BUILTIN_COLUMNS = {
    "run_name",
    "run_dir",
    "status",
    "best_step",
    "primary_metric_value",
    "selected_metrics_csv",
}


def build_flat_table(
    summaries: Sequence[RunSummary],
    metrics: Sequence[str],
    param_aliases: Sequence[ParamAlias],
    primary_metric: str,
    columns: Sequence[str] | None,
    identifier_columns: Sequence[str] | None,
    order_by_specs: Sequence[str],
    sort_metric_spec: str | None,
    metric_directions: Dict[str, str],
) -> FlatTable:
    """Build a sorted flat table from run summaries."""
    requested_columns = list(columns) if columns else _default_columns(
        metrics=metrics,
        param_aliases=param_aliases,
    )
    _validate_columns(requested_columns, metrics, param_aliases)
    requested_identifier_columns = _validate_identifier_columns(
        identifier_columns or [],
        requested_columns,
    )
    order_by = _parse_order_by_specs(order_by_specs)
    sort_metric = _parse_sort_metric_spec(sort_metric_spec)
    _validate_sort_modes(order_by, sort_metric)
    _validate_order_by_columns(order_by, param_aliases)
    if sort_metric is not None:
        _validate_sort_metric(sort_metric[0], summaries)

    row_metrics = list(metrics)
    if sort_metric is not None and sort_metric[0] not in row_metrics:
        row_metrics.append(sort_metric[0])
    rows = [
        _build_row(
            summary=summary,
            metrics=row_metrics,
            primary_metric=primary_metric,
        )
        for summary in summaries
    ]
    rows = _sort_rows(rows, order_by=order_by, sort_metric=sort_metric)
    highlights = _highlight_best_metric_rows(
        rows=rows,
        columns=requested_columns,
        metric_directions=metric_directions,
    )
    return FlatTable(
        columns=requested_columns,
        rows=rows,
        identifier_columns=requested_identifier_columns,
        highlighted_columns=highlights,
    )


def _default_columns(
    metrics: Sequence[str],
    param_aliases: Sequence[ParamAlias],
) -> list[str]:
    return [
        "run_name",
        *[alias.alias for alias in param_aliases],
        "best_step",
        *metrics,
        "status",
    ]


def _validate_columns(
    columns: Sequence[str],
    metrics: Sequence[str],
    param_aliases: Sequence[ParamAlias],
) -> None:
    param_names = {alias.alias for alias in param_aliases}
    metric_names = set(metrics)
    allowed = BUILTIN_COLUMNS | param_names | metric_names
    unknown = [column for column in columns if column not in allowed]
    if unknown:
        raise ValueError(
            "Unknown table column(s): "
            f"{', '.join(unknown)}. Allowed: {', '.join(sorted(allowed))}."
        )


def _validate_identifier_columns(
    identifier_columns: Sequence[str],
    requested_columns: Sequence[str],
) -> list[str]:
    requested = set(requested_columns)
    unknown = [column for column in identifier_columns if column not in requested]
    if unknown:
        raise ValueError(
            "Identifier column(s) must also be present in --columns: "
            f"{', '.join(unknown)}."
        )

    expected_prefix = list(requested_columns[: len(identifier_columns)])
    if list(identifier_columns) != expected_prefix:
        raise ValueError(
            "Identifier columns must be the leading table columns. "
            f"Expected prefix: {', '.join(identifier_columns)}."
        )
    return list(identifier_columns)


def _build_row(
    summary: RunSummary,
    metrics: Sequence[str],
    primary_metric: str,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "run_name": summary.run_name,
        "run_dir": str(summary.run_dir),
        "status": summary.status,
        "best_step": summary.best_step,
        "primary_metric_value": summary.metrics.get(primary_metric),
        "selected_metrics_csv": _path_to_string(summary.selected_metrics_csv),
    }
    for name, value in summary.params.items():
        row[name] = value
    for metric_name in metrics:
        row[metric_name] = summary.metrics.get(metric_name)
    return row


def _path_to_string(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path)


def _sort_rows(
    rows: list[Dict[str, Any]],
    order_by: Sequence[tuple[str, bool]],
    sort_metric: tuple[str, bool] | None,
) -> list[Dict[str, Any]]:
    if sort_metric is not None:
        key_name, descending = sort_metric
        return _sort_rows_by_single_key(rows, key_name=key_name, descending=descending)

    if not order_by:
        return rows

    sorted_rows = list(rows)
    for key_name, descending in reversed(order_by):
        sorted_rows = _sort_rows_by_single_key(
            sorted_rows,
            key_name=key_name,
            descending=descending,
        )
    return sorted_rows


def _sort_rows_by_single_key(
    rows: list[Dict[str, Any]],
    key_name: str,
    descending: bool,
) -> list[Dict[str, Any]]:
    present = [row for row in rows if _has_sort_value(row.get(key_name))]
    missing = [row for row in rows if not _has_sort_value(row.get(key_name))]
    return sorted(
        present,
        key=lambda row: _sort_value(row.get(key_name)),
        reverse=descending,
    ) + missing


def _parse_order_by_specs(order_by_specs: Sequence[str]) -> list[tuple[str, bool]]:
    parsed: list[tuple[str, bool]] = []
    for spec_group in order_by_specs:
        for spec in spec_group.split(","):
            spec = spec.strip()
            if not spec:
                continue
            parsed.append(_parse_column_sort_spec(spec, flag_name="--order-by"))
    return parsed


def _parse_sort_metric_spec(sort_metric_spec: str | None) -> tuple[str, bool] | None:
    if sort_metric_spec is None:
        return None
    return _parse_metric_sort_spec(sort_metric_spec, flag_name="--sort-metric")


def _parse_column_sort_spec(spec: str, flag_name: str) -> tuple[str, bool]:
    parts = [part.strip() for part in spec.split(":") if part.strip()]
    if len(parts) == 2:
        key_name, direction = parts
    elif len(parts) == 3 and parts[0] in {"param", "column", "identifier"}:
        _, key_name, direction = parts
    else:
        raise ValueError(
            f"Invalid {flag_name} value '{spec}'. Expected column:asc|desc, "
            "param:alias:asc|desc, or identifier:alias:asc|desc."
        )
    return key_name, _parse_descending(direction)


def _parse_metric_sort_spec(spec: str, flag_name: str) -> tuple[str, bool]:
    parts = [part.strip() for part in spec.split(":") if part.strip()]
    if len(parts) == 2:
        metric_name, direction = parts
    elif len(parts) == 3 and parts[0] == "metric":
        _, metric_name, direction = parts
    else:
        raise ValueError(
            f"Invalid {flag_name} value '{spec}'. Expected metric:asc|desc "
            "or metric:name:asc|desc."
        )
    return metric_name, _parse_descending(direction)


def _parse_descending(direction: str) -> bool:
    direction = direction.lower()
    if direction not in {"asc", "desc"}:
        raise ValueError(f"Sort direction must be asc or desc, got '{direction}'.")
    return direction == "desc"


def _validate_sort_modes(
    order_by: Sequence[tuple[str, bool]],
    sort_metric: tuple[str, bool] | None,
) -> None:
    if order_by and sort_metric is not None:
        raise ValueError("--order-by and --sort-metric are mutually exclusive.")


def _validate_order_by_columns(
    order_by: Sequence[tuple[str, bool]],
    param_aliases: Sequence[ParamAlias],
) -> None:
    param_names = {alias.alias for alias in param_aliases}
    allowed = BUILTIN_COLUMNS | param_names
    unknown = [column for column, _ in order_by if column not in allowed]
    if unknown:
        raise ValueError(
            "Unknown --order-by column(s): "
            f"{', '.join(unknown)}. Allowed: {', '.join(sorted(allowed))}."
        )


def _validate_sort_metric(metric_name: str, summaries: Sequence[RunSummary]) -> None:
    if any(metric_name in summary.metrics for summary in summaries):
        return
    raise ValueError(f"Metric '{metric_name}' was not found in any summarized run.")


def _has_sort_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return True


def _sort_value(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    return str(value)


def _highlight_best_metric_rows(
    rows: Sequence[Dict[str, Any]],
    columns: Sequence[str],
    metric_directions: Dict[str, str],
) -> Dict[str, set[int]]:
    highlights: Dict[str, set[int]] = {}
    for column in columns:
        numeric_values: list[tuple[int, float]] = []
        for idx, row in enumerate(rows):
            value = row.get(column)
            if isinstance(value, (int, float)):
                numeric = float(value)
                if not math.isnan(numeric):
                    numeric_values.append((idx, numeric))
        if not numeric_values:
            continue
        if column not in metric_directions:
            continue
        direction = get_metric_direction(column, metric_directions)
        best_idx, best_value = numeric_values[0]
        best_indices = {best_idx}
        for idx, value in numeric_values[1:]:
            if is_better(value, best_value, direction):
                best_value = value
                best_indices = {idx}
            elif value == best_value:
                best_indices.add(idx)
        highlights[column] = best_indices
    return highlights

