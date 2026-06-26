"""Read Hydra run artifacts and best-checkpoint metrics CSVs."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List

from omegaconf import DictConfig, OmegaConf

from scripts.reporting.metric_selection import select_best_snapshot
from scripts.reporting.schema import MetricSnapshot, RunSummary


STEP_PATTERN = re.compile(r"best_model_step_(\d+)_")


def load_run_config(run_dir: Path) -> DictConfig:
    """Load the resolved Hydra config for a run directory."""
    config_path = Path(run_dir) / ".hydra" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing Hydra config: {config_path}")
    return OmegaConf.load(config_path)


def read_metrics_csv(path: Path) -> Dict[str, float]:
    """Read a checkpoint metrics CSV with ``metric_key,metric_value`` schema."""
    metrics: Dict[str, float] = {}
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected = {"metric_key", "metric_value"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError(
                f"Metrics CSV must have columns metric_key,metric_value: {path}"
            )
        for row in reader:
            key = str(row["metric_key"]).strip()
            if not key:
                continue
            metrics[key] = float(row["metric_value"])
    if not metrics:
        raise ValueError(f"Metrics CSV contains no metric rows: {path}")
    return metrics


def find_metric_snapshots(run_dir: Path) -> List[MetricSnapshot]:
    """Find and parse all retained best-checkpoint metrics CSVs for a run."""
    best_dir = Path(run_dir) / "models" / "best"
    if not best_dir.is_dir():
        return []

    snapshots: List[MetricSnapshot] = []
    for csv_path in sorted(best_dir.glob("*_metrics.csv")):
        metrics = read_metrics_csv(csv_path)
        snapshots.append(
            MetricSnapshot(
                path=csv_path,
                step=extract_step_from_metrics_path(csv_path),
                metrics=metrics,
            )
        )
    return snapshots


def extract_step_from_metrics_path(path: Path) -> int | None:
    """Extract checkpoint step from a best metrics filename."""
    match = STEP_PATTERN.search(Path(path).name)
    if match is None:
        return None
    return int(match.group(1))


def summarize_run(
    run_dir: Path,
    primary_metric: str,
    primary_direction: str,
) -> RunSummary:
    """Build a ``RunSummary`` with the selected best-checkpoint metrics."""
    run_dir = Path(run_dir)
    summary = RunSummary(
        run_name=run_dir.name,
        run_dir=run_dir,
        status="complete",
        config_path=run_dir / ".hydra" / "config.yaml",
        overrides_path=run_dir / ".hydra" / "overrides.yaml",
    )

    try:
        snapshots = find_metric_snapshots(run_dir)
    except (OSError, ValueError) as exc:
        summary.status = "invalid_metrics_csv"
        summary.errors.append(str(exc))
        return summary

    if not snapshots:
        summary.status = "no_best_metrics"
        summary.errors.append(f"No best metrics CSVs found under {run_dir / 'models' / 'best'}")
        return summary

    try:
        selected = select_best_snapshot(
            snapshots=snapshots,
            primary_metric=primary_metric,
            primary_direction=primary_direction,
        )
    except KeyError as exc:
        summary.status = "missing_primary_metric"
        summary.errors.append(str(exc))
        return summary
    except ValueError as exc:
        summary.status = "invalid_primary_metric"
        summary.errors.append(str(exc))
        return summary

    summary.selected_snapshot = selected
    summary.metrics = dict(selected.metrics)
    return summary

