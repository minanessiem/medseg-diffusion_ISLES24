"""Summarize existing experiment-holder directories.

Example:
    python3 -m scripts.reporting.summarize \
      --experiment-dir outputs/example_experiment \
      --primary-metric dice_3d \
      --primary-direction max \
      --metrics dice_3d,surface_dice_monai_3d,hd95_3d
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from scripts.reporting.config_projection import (
    attach_projected_params,
    parse_param_aliases,
)
from scripts.reporting.experiment_discovery import discover_run_dirs
from scripts.reporting.metric_directions import build_metric_directions
from scripts.reporting.metric_selection import validate_direction
from scripts.reporting.pivot_table import build_grid_tables
from scripts.reporting.provenance import build_reporting_provenance
from scripts.reporting.renderers import (
    parse_highlight_styles,
    render_flat_table,
    render_grid_tables,
)
from scripts.reporting.run_reader import summarize_run
from scripts.reporting.schema import ParamAlias, RunSummary
from scripts.reporting.summary_table import build_flat_table


SUPPORTED_FORMATS = ("console", "markdown", "csv", "json", "latex")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Summarize runs in an existing experiment-holder directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--experiment-dir",
        required=True,
        type=Path,
        help="Directory whose direct children are Hydra run directories.",
    )
    parser.add_argument(
        "--primary-metric",
        required=True,
        help="Metric used to select one best-checkpoint metrics CSV per run.",
    )
    parser.add_argument(
        "--primary-direction",
        required=True,
        choices=("max", "min"),
        help="Whether higher or lower primary metric values are better.",
    )
    parser.add_argument(
        "--metrics",
        default=None,
        help="Comma-separated metric columns to include. Defaults to primary metric.",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        metavar="ALIAS=CONFIG_PATH",
        help="Config parameter alias, repeatable.",
    )
    parser.add_argument(
        "--columns",
        default=None,
        help="Comma-separated flat-table columns.",
    )
    parser.add_argument(
        "--identifier-columns",
        default=None,
        help=(
            "Comma-separated leading flat-table columns to treat as identifiers. "
            "Human-readable renderers may visually separate them from result columns."
        ),
    )
    parser.add_argument(
        "--order-by",
        action="append",
        default=[],
        help=(
            "Identifier/config ordering spec, repeatable or comma-separated. "
            "Examples: roi:asc,pos:desc or run_name:asc. Mutually exclusive "
            "with --sort-metric."
        ),
    )
    parser.add_argument(
        "--sort-metric",
        default=None,
        help=(
            "Metric ranking spec, e.g. dice_3d:desc or hd95_3d:asc. "
            "Mutually exclusive with --order-by."
        ),
    )
    parser.add_argument(
        "--grid-rows",
        default=None,
        help="Parameter alias to use as grid rows.",
    )
    parser.add_argument(
        "--grid-cols",
        default=None,
        help="Parameter alias to use as grid columns.",
    )
    parser.add_argument(
        "--grid-values",
        default=None,
        help="Comma-separated metrics to render as grid tables.",
    )
    parser.add_argument(
        "--metric-direction",
        action="append",
        default=[],
        metavar="METRIC=max|min",
        help="Override metric direction for highlighting, repeatable.",
    )
    parser.add_argument(
        "--format",
        default="console",
        choices=SUPPORTED_FORMATS,
        help="Output format.",
    )
    parser.add_argument(
        "--highlight-best",
        default="none",
        help=(
            "Comma-separated styling for best metric values in human-readable "
            "formats: none, bold, underline, or bold,underline."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file. Defaults to stdout.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)

    try:
        output = summarize_from_args(args, argv=raw_argv)
    except Exception as exc:  # noqa: BLE001 - CLI should print concise errors.
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output)
    return 0


def summarize_from_args(args: argparse.Namespace, argv: Sequence[str]) -> str:
    """Build and render the requested summary."""
    primary_direction = validate_direction(args.primary_direction)
    metrics = _parse_csv_list(args.metrics) or [args.primary_metric]
    param_aliases = parse_param_aliases(args.param)
    highlight_styles = parse_highlight_styles(args.highlight_best)
    metric_directions = build_metric_directions(
        specs=args.metric_direction,
        primary_metric=args.primary_metric,
        primary_direction=primary_direction,
    )
    provenance = build_reporting_provenance(args, argv)

    summaries = _load_summaries(
        experiment_dir=args.experiment_dir,
        primary_metric=args.primary_metric,
        primary_direction=primary_direction,
        param_aliases=param_aliases,
    )

    if args.grid_rows or args.grid_cols or args.grid_values:
        _validate_grid_args(args)
        grid_values = _parse_csv_list(args.grid_values)
        tables = build_grid_tables(
            summaries=summaries,
            row_alias=args.grid_rows,
            col_alias=args.grid_cols,
            metric_names=grid_values,
            param_aliases=param_aliases,
            metric_directions=metric_directions,
        )
        return render_grid_tables(
            tables,
            args.format,
            highlight_styles=highlight_styles,
            provenance=provenance,
        )

    table = build_flat_table(
        summaries=summaries,
        metrics=metrics,
        param_aliases=param_aliases,
        primary_metric=args.primary_metric,
        columns=_parse_csv_list(args.columns),
        identifier_columns=_parse_csv_list(args.identifier_columns),
        order_by_specs=args.order_by,
        sort_metric_spec=args.sort_metric,
        metric_directions=metric_directions,
    )
    return render_flat_table(
        table,
        args.format,
        highlight_styles=highlight_styles,
        provenance=provenance,
    )


def _load_summaries(
    experiment_dir: Path,
    primary_metric: str,
    primary_direction: str,
    param_aliases: Sequence[ParamAlias],
) -> list[RunSummary]:
    run_dirs = discover_run_dirs(experiment_dir)
    summaries = [
        summarize_run(
            run_dir=run_dir,
            primary_metric=primary_metric,
            primary_direction=primary_direction,
        )
        for run_dir in run_dirs
    ]
    attach_projected_params(summaries, param_aliases)
    return summaries


def _parse_csv_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _validate_grid_args(args: argparse.Namespace) -> None:
    missing = []
    if not args.grid_rows:
        missing.append("--grid-rows")
    if not args.grid_cols:
        missing.append("--grid-cols")
    if not args.grid_values:
        missing.append("--grid-values")
    if missing:
        raise ValueError(
            "Grid output requires all grid arguments: " + ", ".join(missing)
        )


if __name__ == "__main__":
    raise SystemExit(main())

