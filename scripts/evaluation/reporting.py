"""
Canonical reporting helpers for greenfield evaluation.

This module converts finalized metrics-engine output into a stable report
payload and file outputs (JSON/CSV).
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from scripts.evaluation.contracts import ThresholdProtocol


def build_report_payload(
    finalized_results: Dict[float, Dict[str, object]],
    protocol: ThresholdProtocol,
    entrypoint_name: str,
    metadata: Optional[Dict[str, object]] = None,
    selected_threshold: Optional[float] = None,
    auc: Optional[Dict[str, float]] = None,
    volume_finalized_results: Optional[Dict[float, Dict[str, object]]] = None,
) -> Dict[str, object]:
    """
    Build canonical report payload from finalized evaluation results.

    Args:
        finalized_results: Output from StreamingMetricsEngine.finalize()
        protocol: Threshold protocol used by the run
        entrypoint_name: Name of entrypoint script
        metadata: Optional metadata fields merged into metadata block
        selected_threshold: Primary threshold for summary reporting
        auc: Optional AUC metrics (probability-based paths)
    """
    if not finalized_results:
        raise ValueError("finalized_results must not be empty.")

    ordered_thresholds = sorted(float(t) for t in finalized_results.keys())
    first_threshold = ordered_thresholds[0]
    first_result = finalized_results[first_threshold]
    slice_counts = dict(first_result["slice_counts"])

    metric_names = _infer_metric_names(finalized_results)
    threshold_rows = [finalized_results[t] for t in ordered_thresholds]

    if selected_threshold is None and protocol.mode == "fixed":
        selected_threshold = float(protocol.thresholds[0])

    payload = {
        "metadata": {
            "entrypoint": entrypoint_name,
            "analysis_timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            **(metadata or {}),
        },
        "data_summary": {
            "total_slices": int(slice_counts["total"]),
            "foreground_slices": int(slice_counts["foreground"]),
            "empty_slices": int(slice_counts["empty"]),
        },
        "protocol": {
            "mode": protocol.mode,
            "thresholds_evaluated": [float(t) for t in protocol.thresholds],
            "optimize_metric": protocol.optimize_metric,
            "selected_threshold": selected_threshold,
            "scope_note": (
                "All metrics report all_slices and foreground_only. "
                "Dice foreground semantics come from metric implementation."
            ),
        },
        "metrics": {
            "metric_names": metric_names,
            "threshold_results": threshold_rows,
        },
    }
    payload["metrics"]["slice_level"] = {
        "metric_names": metric_names,
        "threshold_results": threshold_rows,
    }

    if selected_threshold is not None:
        payload["metrics"]["primary_metrics_at_selected_threshold"] = (
            finalized_results[float(selected_threshold)]["metrics"]
        )

    default_threshold = 0.5
    default_threshold_key = _lookup_threshold_key(
        finalized_results=finalized_results,
        target_threshold=default_threshold,
    )
    if default_threshold_key is not None:
        payload["metrics"]["default_threshold_metrics"] = {
            "requested_threshold": default_threshold,
            "evaluated": True,
            "evaluated_threshold": float(default_threshold_key),
            "metrics": finalized_results[default_threshold_key]["metrics"],
        }
    else:
        payload["metrics"]["default_threshold_metrics"] = {
            "requested_threshold": default_threshold,
            "evaluated": False,
            "evaluated_threshold": None,
            "metrics": None,
        }

    if auc is not None:
        payload["auc"] = {
            "roc": float(auc.get("roc", 0.0)),
            "pr": float(auc.get("pr", 0.0)),
        }

    if volume_finalized_results:
        volume_metric_names = _infer_metric_names_single_scope(volume_finalized_results)
        volume_threshold_rows = [
            volume_finalized_results[t] for t in sorted(float(t) for t in volume_finalized_results.keys())
        ]
        payload["metrics"]["volume_level"] = {
            "metric_names": volume_metric_names,
            "threshold_results": volume_threshold_rows,
        }
        first_volume = volume_threshold_rows[0]
        volume_counts = dict(first_volume.get("volume_counts", {}))
        volume_slice_counts = dict(first_volume.get("volume_slice_counts", {}))
        payload["data_summary"].update(
            {
                "total_volumes": int(volume_counts.get("total", 0)),
                "foreground_volumes": int(volume_counts.get("foreground", 0)),
                "empty_volumes": int(volume_counts.get("empty", 0)),
                "volume_slice_count_mean": float(volume_slice_counts.get("mean", 0.0)),
                "volume_slice_count_std": float(volume_slice_counts.get("std", 0.0)),
                "volume_slice_count_min": int(volume_slice_counts.get("min", 0)),
                "volume_slice_count_max": int(volume_slice_counts.get("max", 0)),
                "volume_slice_count_total": int(volume_slice_counts.get("total", 0)),
            }
        )
        if selected_threshold is not None and float(selected_threshold) in volume_finalized_results:
            payload["metrics"]["volume_level"]["primary_metrics_at_selected_threshold"] = (
                volume_finalized_results[float(selected_threshold)]["metrics"]
            )

    return payload


def write_json_report(
    payload: Dict[str, object],
    output_dir: Path,
    filename: str = "canonical_results.json",
) -> Path:
    """Write canonical payload as JSON and return output path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def write_threshold_csv(
    finalized_results: Dict[float, Dict[str, object]],
    output_dir: Path,
    filename: str = "metrics_per_threshold.csv",
    metric_names: Optional[Sequence[str]] = None,
) -> Path:
    """
    Write flattened metrics-per-threshold CSV.

    Column format includes:
    - threshold and slice counts
    - per-metric all_slices and foreground_only stats
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    if metric_names is None:
        metric_names = _infer_metric_names(finalized_results)

    header = ["threshold", "total_slices", "foreground_slices", "empty_slices"]
    for metric_name in metric_names:
        header.extend(
            [
                f"{metric_name}_all_mean",
                f"{metric_name}_all_std",
                f"{metric_name}_all_count",
                f"{metric_name}_fg_mean",
                f"{metric_name}_fg_std",
                f"{metric_name}_fg_count",
            ]
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)

        for threshold in sorted(float(t) for t in finalized_results.keys()):
            row = _flatten_threshold_row(finalized_results[threshold], threshold, metric_names)
            writer.writerow(row)

    return path


def write_volume_threshold_csv(
    finalized_results: Dict[float, Dict[str, object]],
    output_dir: Path,
    filename: str = "volume_metrics_per_threshold.csv",
    metric_names: Optional[Sequence[str]] = None,
) -> Path:
    """
    Write flattened volume-level metrics-per-threshold CSV (single-scope metrics).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    if metric_names is None:
        metric_names = _infer_metric_names_single_scope(finalized_results)
    header = [
        "threshold",
        "total_volumes",
        "foreground_volumes",
        "empty_volumes",
        "volume_slice_count_mean",
        "volume_slice_count_std",
        "volume_slice_count_min",
        "volume_slice_count_max",
        "volume_slice_count_total",
    ]
    for metric_name in metric_names:
        header.extend([f"{metric_name}_mean", f"{metric_name}_std", f"{metric_name}_count"])

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for threshold in sorted(float(t) for t in finalized_results.keys()):
            result = finalized_results[threshold]
            counts = result.get("volume_counts", {})
            slice_counts = result.get("volume_slice_counts", {})
            row: List[object] = [
                float(threshold),
                int(counts.get("total", 0)),
                int(counts.get("foreground", 0)),
                int(counts.get("empty", 0)),
                float(slice_counts.get("mean", 0.0)),
                float(slice_counts.get("std", 0.0)),
                int(slice_counts.get("min", 0)),
                int(slice_counts.get("max", 0)),
                int(slice_counts.get("total", 0)),
            ]
            metrics = result.get("metrics", {})
            for metric_name in metric_names:
                stats = metrics.get(metric_name, {})
                row.extend(
                    [
                        float(stats.get("mean", 0.0)),
                        float(stats.get("std", 0.0)),
                        int(stats.get("count", 0)),
                    ]
                )
            writer.writerow(row)
    return path


def append_per_slice_metrics_rows(
    rows: Iterable[Dict[str, object]],
    output_csv_path: Path,
    fieldnames: Sequence[str],
) -> Path:
    """
    Append per-slice metric rows to CSV in stream-friendly mode.

    If file does not exist yet, header is written first.
    """
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not output_csv_path.exists()

    with output_csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        if should_write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_csv_path


def build_text_summary(payload: Dict[str, object]) -> str:
    """Build a concise human-readable summary from canonical payload."""
    meta = payload["metadata"]
    data_summary = payload["data_summary"]
    protocol = payload["protocol"]
    lines = [
        "Unified Segmentation Evaluation Summary",
        "=" * 50,
        f"Entrypoint: {meta['entrypoint']}",
        f"Timestamp: {meta['analysis_timestamp']}",
        "",
        "Slice counts:",
        f"  Total:      {data_summary['total_slices']}",
        f"  Foreground: {data_summary['foreground_slices']}",
        f"  Empty:      {data_summary['empty_slices']}",
        "",
        "Protocol:",
        f"  Mode:               {protocol['mode']}",
        f"  Thresholds:         {protocol['thresholds_evaluated']}",
        f"  Optimize metric:    {protocol['optimize_metric']}",
        f"  Selected threshold: {protocol['selected_threshold']}",
        "",
    ]
    if "total_volumes" in data_summary:
        lines.extend(
            [
                "Volume counts:",
                f"  Total:      {data_summary['total_volumes']}",
                f"  Foreground: {data_summary['foreground_volumes']}",
                f"  Empty:      {data_summary['empty_volumes']}",
                "Volume slice counts:",
                f"  Mean:       {data_summary['volume_slice_count_mean']:.2f}",
                f"  Std:        {data_summary['volume_slice_count_std']:.2f}",
                f"  Min:        {data_summary['volume_slice_count_min']}",
                f"  Max:        {data_summary['volume_slice_count_max']}",
                "",
            ]
        )
    if "auc" in payload:
        lines.extend(
            [
                "AUC:",
                f"  ROC: {payload['auc']['roc']:.4f}",
                f"  PR:  {payload['auc']['pr']:.4f}",
                "",
            ]
        )
    lines.append("=" * 50)
    return "\n".join(lines)


def _flatten_threshold_row(
    threshold_result: Dict[str, object],
    threshold: float,
    metric_names: Sequence[str],
) -> List[object]:
    slice_counts = threshold_result["slice_counts"]
    metrics = threshold_result["metrics"]
    row: List[object] = [
        float(threshold),
        int(slice_counts["total"]),
        int(slice_counts["foreground"]),
        int(slice_counts["empty"]),
    ]

    for metric_name in metric_names:
        metric_scopes = metrics[metric_name]
        all_stats = metric_scopes["all_slices"]
        fg_stats = metric_scopes["foreground_only"]
        row.extend(
            [
                float(all_stats["mean"]),
                float(all_stats["std"]),
                int(all_stats["count"]),
                float(fg_stats["mean"]),
                float(fg_stats["std"]),
                int(fg_stats["count"]),
            ]
        )
    return row


def _infer_metric_names(finalized_results: Dict[float, Dict[str, object]]) -> List[str]:
    first_threshold = sorted(float(t) for t in finalized_results.keys())[0]
    metric_map = finalized_results[first_threshold]["metrics"]
    return list(metric_map.keys())


def _infer_metric_names_single_scope(finalized_results: Dict[float, Dict[str, object]]) -> List[str]:
    first_threshold = sorted(float(t) for t in finalized_results.keys())[0]
    metric_map = finalized_results[first_threshold]["metrics"]
    return list(metric_map.keys())


def _lookup_threshold_key(
    finalized_results: Dict[float, Dict[str, object]],
    target_threshold: float,
    tol: float = 1e-9,
) -> Optional[float]:
    for threshold in finalized_results.keys():
        if abs(float(threshold) - float(target_threshold)) <= tol:
            return float(threshold)
    return None

