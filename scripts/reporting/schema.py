"""Shared data contracts for experiment reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass(frozen=True)
class ParamAlias:
    """Named projection from a Hydra config path."""

    alias: str
    config_path: str


@dataclass(frozen=True)
class MetricSnapshot:
    """Metrics captured for one retained best checkpoint."""

    path: Path
    step: Optional[int]
    metrics: Dict[str, float]


@dataclass
class RunSummary:
    """One summarized run within an experiment-holder directory."""

    run_name: str
    run_dir: Path
    status: str
    config_path: Optional[Path] = None
    overrides_path: Optional[Path] = None
    selected_snapshot: Optional[MetricSnapshot] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    @property
    def best_step(self) -> Optional[int]:
        """Return the selected checkpoint step, if available."""
        if self.selected_snapshot is None:
            return None
        return self.selected_snapshot.step

    @property
    def selected_metrics_csv(self) -> Optional[Path]:
        """Return the selected metrics CSV path, if available."""
        if self.selected_snapshot is None:
            return None
        return self.selected_snapshot.path


@dataclass(frozen=True)
class FlatTable:
    """A display-ready flat table."""

    columns: Sequence[str]
    rows: List[Dict[str, Any]]
    identifier_columns: Sequence[str] = field(default_factory=tuple)
    highlighted_columns: Dict[str, set[int]] = field(default_factory=dict)


@dataclass(frozen=True)
class GridTable:
    """A display-ready pivot table for one metric."""

    metric_name: str
    row_alias: str
    col_alias: str
    row_values: Sequence[Any]
    col_values: Sequence[Any]
    cells: Dict[tuple[Any, Any], Any]
    highlighted_cells: set[tuple[Any, Any]] = field(default_factory=set)


@dataclass(frozen=True)
class SummaryRequest:
    """Resolved request for summarizing an experiment directory."""

    experiment_dir: Path
    primary_metric: str
    primary_direction: str
    metrics: Sequence[str]
    param_aliases: Sequence[ParamAlias]
    columns: Optional[Sequence[str]] = None
    identifier_columns: Optional[Sequence[str]] = None
    order_by_specs: Sequence[str] = field(default_factory=tuple)
    sort_metric_spec: Optional[str] = None
    grid_rows: Optional[str] = None
    grid_cols: Optional[str] = None
    grid_values: Optional[Sequence[str]] = None
    metric_directions: Dict[str, str] = field(default_factory=dict)

