"""
Per-case threshold records and aggregate helpers for model evaluation.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.evaluation.core.contracts import PrimaryMetricSelector
from scripts.evaluation.reporting.threshold_selection import select_best_threshold_from_rows


PREDICTED_VOLUME_KEY = "PredictedVolumeMm3"
GROUND_TRUTH_VOLUME_KEY = "GroundTruthVolumeMm3"
VOLUME_RATIO_KEY = "pred_gt_volume_ratio"


@dataclass(frozen=True)
class ThresholdMetricRecord:
    """One compact metric row for a case at one threshold and level."""

    level: str
    case_id: str
    threshold: float
    metrics: Mapping[str, float]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def with_metrics(self, metrics: Mapping[str, float]) -> "ThresholdMetricRecord":
        """Return a copy with replaced metrics."""
        return ThresholdMetricRecord(
            level=self.level,
            case_id=self.case_id,
            threshold=float(self.threshold),
            metrics=dict(metrics),
            metadata=dict(self.metadata),
        )


def add_volume_ratio(metrics: Mapping[str, float]) -> Dict[str, float]:
    """
    Add predicted/ground-truth volume ratio when volume metrics are present.
    """
    updated = {str(key): float(value) for key, value in metrics.items()}
    if PREDICTED_VOLUME_KEY not in updated or GROUND_TRUTH_VOLUME_KEY not in updated:
        return updated

    predicted_volume = float(updated[PREDICTED_VOLUME_KEY])
    ground_truth_volume = float(updated[GROUND_TRUTH_VOLUME_KEY])
    if ground_truth_volume == 0.0:
        ratio = 1.0 if predicted_volume == 0.0 else math.inf
    else:
        ratio = predicted_volume / ground_truth_volume
    updated[VOLUME_RATIO_KEY] = float(ratio)
    return updated


def aggregate_threshold_records(
    records: Iterable[ThresholdMetricRecord],
    selector_level: str,
) -> Dict[float, Dict[str, object]]:
    """
    Aggregate records by threshold for one evaluation level.
    """
    grouped: Dict[float, List[ThresholdMetricRecord]] = defaultdict(list)
    for record in records:
        if record.level != selector_level:
            continue
        grouped[float(record.threshold)].append(record)

    aggregates: Dict[float, Dict[str, object]] = {}
    for threshold in sorted(grouped):
        threshold_records = grouped[threshold]
        metric_values: Dict[str, List[float]] = defaultdict(list)
        for record in threshold_records:
            for metric_name, value in record.metrics.items():
                value_float = float(value)
                if math.isnan(value_float):
                    continue
                metric_values[str(metric_name)].append(value_float)

        aggregates[threshold] = {
            "level": selector_level,
            "threshold": float(threshold),
            "case_count": len({record.case_id for record in threshold_records}),
            "record_count": len(threshold_records),
            "metrics": {
                metric_name: _compute_stats(values)
                for metric_name, values in sorted(metric_values.items())
            },
        }

    return aggregates


def select_global_threshold(
    records: Iterable[ThresholdMetricRecord],
    selector: PrimaryMetricSelector,
) -> Dict[str, object]:
    """
    Select the best global threshold from per-case threshold records.
    """
    aggregates = aggregate_threshold_records(records, selector_level=selector.level)
    rows = list(aggregates.values())
    selected_threshold = select_best_threshold_from_rows(rows, selector)
    selected_row = aggregates[float(selected_threshold)]
    selected_metric = selected_row["metrics"][selector.metric]
    selected_value = float(selected_metric[selector.statistic])
    return {
        "threshold": float(selected_threshold),
        "selector": _selector_to_dict(selector),
        "selected_statistic_value": selected_value,
        "metrics": selected_row["metrics"],
        "aggregate_row": selected_row,
    }


def select_oracle_thresholds(
    records: Iterable[ThresholdMetricRecord],
    selector: PrimaryMetricSelector,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """
    Select each case's best threshold according to ``selector``.
    """
    by_case: Dict[str, List[ThresholdMetricRecord]] = defaultdict(list)
    for record in records:
        if record.level == selector.level:
            by_case[str(record.case_id)].append(record)

    oracle_rows: List[Dict[str, object]] = []
    selected_records: List[ThresholdMetricRecord] = []
    for case_id in sorted(by_case):
        selected = _select_best_record_for_case(by_case[case_id], selector)
        selected_records.append(selected)
        selected_value = float(selected.metrics[selector.metric])
        oracle_rows.append(
            {
                "level": selected.level,
                "case_id": selected.case_id,
                "threshold": float(selected.threshold),
                "selected_metric": selector.metric,
                "selected_statistic": selector.statistic,
                "selected_value": selected_value,
                "metrics": dict(selected.metrics),
                "metadata": dict(selected.metadata),
            }
        )

    metric_values: Dict[str, List[float]] = defaultdict(list)
    for record in selected_records:
        for metric_name, value in record.metrics.items():
            value_float = float(value)
            if not math.isnan(value_float):
                metric_values[str(metric_name)].append(value_float)

    threshold_counts = Counter(float(record.threshold) for record in selected_records)
    summary = {
        "selector": _selector_to_dict(selector),
        "case_count": len(selected_records),
        "threshold_counts": {
            str(threshold): int(count)
            for threshold, count in sorted(threshold_counts.items())
        },
        "selected_thresholds": _compute_stats(
            [float(record.threshold) for record in selected_records]
        )
        if selected_records
        else _empty_stats(),
        "metrics": {
            metric_name: _compute_stats(values)
            for metric_name, values in sorted(metric_values.items())
        },
    }
    return oracle_rows, summary


def write_per_case_threshold_csv(
    records: Iterable[ThresholdMetricRecord],
    output_dir: Path,
    filename: str = "per_case_threshold_metrics.csv",
) -> Path:
    """
    Write per-case/per-threshold metric records with dynamic metric columns.
    """
    record_list = list(records)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    metric_names = _collect_metric_names(record_list)
    metadata_names = _collect_metadata_names(record_list)
    fieldnames = [
        "level",
        "case_id",
        "threshold",
        *metric_names,
        *[f"metadata.{name}" for name in metadata_names],
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in record_list:
            row = {
                "level": record.level,
                "case_id": record.case_id,
                "threshold": float(record.threshold),
            }
            for metric_name in metric_names:
                row[metric_name] = _format_csv_value(record.metrics.get(metric_name, ""))
            for metadata_name in metadata_names:
                row[f"metadata.{metadata_name}"] = _format_csv_value(
                    record.metadata.get(metadata_name, "")
                )
            writer.writerow(row)

    return path


def write_oracle_threshold_csv(
    rows: Iterable[Mapping[str, object]],
    output_dir: Path,
    filename: str = "oracle_per_case_thresholds.csv",
) -> Path:
    """
    Write per-case oracle threshold selections.
    """
    row_list = list(rows)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    metric_names = sorted(
        {
            str(metric_name)
            for row in row_list
            for metric_name in _mapping_or_empty(row.get("metrics")).keys()
        }
    )
    metadata_names = sorted(
        {
            str(metadata_name)
            for row in row_list
            for metadata_name in _mapping_or_empty(row.get("metadata")).keys()
        }
    )
    fieldnames = [
        "level",
        "case_id",
        "threshold",
        "selected_metric",
        "selected_statistic",
        "selected_value",
        *metric_names,
        *[f"metadata.{name}" for name in metadata_names],
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for source in row_list:
            metrics = _mapping_or_empty(source.get("metrics"))
            metadata = _mapping_or_empty(source.get("metadata"))
            row = {
                "level": source.get("level", ""),
                "case_id": source.get("case_id", ""),
                "threshold": _format_csv_value(source.get("threshold", "")),
                "selected_metric": source.get("selected_metric", ""),
                "selected_statistic": source.get("selected_statistic", ""),
                "selected_value": _format_csv_value(source.get("selected_value", "")),
            }
            for metric_name in metric_names:
                row[metric_name] = _format_csv_value(metrics.get(metric_name, ""))
            for metadata_name in metadata_names:
                row[f"metadata.{metadata_name}"] = _format_csv_value(
                    metadata.get(metadata_name, "")
                )
            writer.writerow(row)

    return path


def _select_best_record_for_case(
    records: Sequence[ThresholdMetricRecord],
    selector: PrimaryMetricSelector,
) -> ThresholdMetricRecord:
    if not records:
        raise ValueError("Cannot select oracle threshold from empty case records.")

    best_record: Optional[ThresholdMetricRecord] = None
    best_value: Optional[float] = None
    for record in sorted(records, key=lambda item: float(item.threshold)):
        if selector.metric not in record.metrics:
            raise ValueError(
                f"Metric '{selector.metric}' missing for case '{record.case_id}' "
                f"at threshold={record.threshold}."
            )
        value = float(record.metrics[selector.metric])
        if math.isnan(value):
            continue
        if best_value is None or _is_better(value, best_value, selector.direction):
            best_value = value
            best_record = record

    if best_record is None:
        raise ValueError("Unable to select oracle threshold from non-finite values.")
    return best_record


def _compute_stats(values: Sequence[float]) -> Dict[str, float]:
    normalized = [float(value) for value in values if not math.isnan(float(value))]
    if not normalized:
        return _empty_stats()

    sorted_values = sorted(normalized)
    count = len(sorted_values)
    mean = sum(sorted_values) / count
    median = _median(sorted_values)
    if count <= 1:
        std = 0.0
    elif all(math.isfinite(value) for value in sorted_values):
        variance = sum((value - mean) ** 2 for value in sorted_values) / count
        std = math.sqrt(max(variance, 0.0))
    else:
        std = math.nan

    return {
        "count": int(count),
        "mean": float(mean),
        "median": float(median),
        "std": float(std),
        "min": float(sorted_values[0]),
        "max": float(sorted_values[-1]),
    }


def _empty_stats() -> Dict[str, float]:
    return {
        "count": 0,
        "mean": math.nan,
        "median": math.nan,
        "std": math.nan,
        "min": math.nan,
        "max": math.nan,
    }


def _median(sorted_values: Sequence[float]) -> float:
    count = len(sorted_values)
    midpoint = count // 2
    if count % 2:
        return float(sorted_values[midpoint])
    return float((sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0)


def _is_better(value: float, best_value: float, direction: str) -> bool:
    if direction == "max":
        return value > best_value
    if direction == "min":
        return value < best_value
    raise ValueError(f"Unsupported selector direction '{direction}'.")


def _selector_to_dict(selector: PrimaryMetricSelector) -> Dict[str, str]:
    return {
        "level": selector.level,
        "metric": selector.metric,
        "statistic": selector.statistic,
        "direction": selector.direction,
    }


def _collect_metric_names(records: Sequence[ThresholdMetricRecord]) -> List[str]:
    return sorted({str(name) for record in records for name in record.metrics.keys()})


def _collect_metadata_names(records: Sequence[ThresholdMetricRecord]) -> List[str]:
    return sorted({str(name) for record in records for name in record.metadata.keys()})


def _mapping_or_empty(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    return {}


def _format_csv_value(value: object) -> object:
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return value
