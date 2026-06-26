"""Reproducibility metadata for reporting outputs."""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence


@dataclass(frozen=True)
class ReportingProvenance:
    """Reproducibility metadata for a rendered reporting artifact."""

    command: str
    parameters: Dict[str, Any]

    def to_lines(self) -> list[str]:
        """Return a compact, human-readable provenance block."""
        lines = [
            "Reporting reproducibility",
            f"command: {self.command}",
            "parameters:",
        ]
        for key, value in self.parameters.items():
            lines.append(f"  {key}: {_format_parameter_value(value)}")
        return lines

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable provenance."""
        return {
            "command": self.command,
            "parameters": self.parameters,
        }


def build_reporting_provenance(args: Any, argv: Sequence[str]) -> ReportingProvenance:
    """Build provenance from parsed CLI args and raw argv."""
    command = "python3 -m scripts.reporting.summarize"
    if argv:
        command = f"{command} {shlex.join(list(argv))}"

    parameters = {
        "experiment_dir": _path_to_string(args.experiment_dir),
        "primary_metric": args.primary_metric,
        "primary_direction": args.primary_direction,
        "metrics": _parse_csv_list(args.metrics) or [args.primary_metric],
        "params": list(args.param),
        "columns": _parse_csv_list(args.columns),
        "identifier_columns": _parse_csv_list(args.identifier_columns),
        "order_by": list(args.order_by),
        "sort_metric": args.sort_metric,
        "grid_rows": args.grid_rows,
        "grid_cols": args.grid_cols,
        "grid_values": _parse_csv_list(args.grid_values),
        "metric_directions": list(args.metric_direction),
        "format": args.format,
        "highlight_best": args.highlight_best,
        "output": _path_to_string(args.output),
    }
    return ReportingProvenance(command=command, parameters=parameters)


def _parse_csv_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _path_to_string(value: Path | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _format_parameter_value(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            return "[]"
        return "[" + ", ".join(str(item) for item in value) + "]"
    if value is None:
        return "null"
    return str(value)

