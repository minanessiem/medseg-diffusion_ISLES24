"""Render reporting tables to human- and machine-readable formats."""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Dict, Sequence

from scripts.reporting.display_names import get_display_name
from scripts.reporting.provenance import ReportingProvenance
from scripts.reporting.schema import FlatTable, GridTable


ANSI_BOLD = "\033[1m"
ANSI_UNDERLINE = "\033[4m"
ANSI_RESET = "\033[0m"
HIGHLIGHT_STYLES = {"bold", "underline"}


def parse_highlight_styles(spec: str | None) -> tuple[str, ...]:
    """Parse comma-separated highlight styles."""
    if spec is None:
        return ()
    styles = tuple(part.strip().lower() for part in spec.split(",") if part.strip())
    if not styles or styles == ("none",):
        return ()
    if "none" in styles:
        raise ValueError("--highlight-best=none cannot be combined with other styles.")
    unknown = [style for style in styles if style not in HIGHLIGHT_STYLES]
    if unknown:
        raise ValueError(
            "Unsupported --highlight-best style(s): "
            f"{', '.join(unknown)}. Allowed: none, bold, underline."
        )
    return styles


def render_flat_table(
    table: FlatTable,
    output_format: str,
    highlight_styles: Sequence[str] = (),
    provenance: ReportingProvenance | None = None,
) -> str:
    """Render a flat table."""
    if output_format == "console":
        return _with_provenance_comment(
            render_flat_console(table, highlight_styles=highlight_styles),
            output_format,
            provenance,
        )
    if output_format == "markdown":
        return _with_provenance_comment(
            render_flat_markdown(table, highlight_styles=highlight_styles),
            output_format,
            provenance,
        )
    if output_format == "csv":
        return _with_provenance_comment(render_flat_csv(table), output_format, provenance)
    if output_format == "json":
        return render_flat_json(table, provenance=provenance)
    if output_format == "latex":
        return _with_provenance_comment(
            render_flat_latex(table, highlight_styles=highlight_styles),
            output_format,
            provenance,
        )
    raise ValueError(f"Unsupported output format for flat table: {output_format}")


def render_grid_tables(
    tables: Sequence[GridTable],
    output_format: str,
    highlight_styles: Sequence[str] = (),
    provenance: ReportingProvenance | None = None,
) -> str:
    """Render one or more grid tables."""
    if output_format == "console":
        return "\n\n".join(
            _with_provenance_comment(
                render_grid_console(table, highlight_styles=highlight_styles),
                output_format,
                provenance,
            )
            for table in tables
        )
    if output_format == "markdown":
        return "\n\n".join(
            _with_provenance_comment(
                render_grid_markdown(table, highlight_styles=highlight_styles),
                output_format,
                provenance,
            )
            for table in tables
        )
    if output_format == "json":
        return render_grid_json(tables, provenance=provenance)
    if output_format == "latex":
        return "\n\n".join(
            _with_provenance_comment(
                render_grid_latex(table, highlight_styles=highlight_styles),
                output_format,
                provenance,
            )
            for table in tables
        )
    if output_format == "csv":
        raise ValueError("CSV grid output is not supported in the MVP.")
    raise ValueError(f"Unsupported output format for grid table: {output_format}")


def render_flat_console(
    table: FlatTable,
    highlight_styles: Sequence[str] = (),
) -> str:
    """Render a flat table as aligned console text."""
    raw_rows = [
        [_format_value(row.get(column)) for column in table.columns]
        for row in table.rows
    ]
    display_columns = [_display_column(column) for column in table.columns]
    widths = _column_widths(display_columns, raw_rows)
    lines = [
        _format_console_row(display_columns, widths),
        _format_console_row(["-" * width for width in widths], widths),
    ]
    for row_idx, raw_row in enumerate(raw_rows):
        styled = []
        for col_idx, raw_value in enumerate(raw_row):
            column = table.columns[col_idx]
            if row_idx in table.highlighted_columns.get(column, set()):
                styled.append(_style_best_console(raw_value, highlight_styles))
            else:
                styled.append(raw_value)
        lines.append(_format_console_row(styled, widths, raw_values=raw_row))
    return "\n".join(lines)


def render_grid_console(
    table: GridTable,
    highlight_styles: Sequence[str] = (),
) -> str:
    """Render a pivot table as aligned console text."""
    columns = [_display_column(table.row_alias), *[str(value) for value in table.col_values]]
    raw_rows = []
    styled_rows = []
    for row_value in table.row_values:
        raw_row = [str(row_value)]
        styled_row = [str(row_value)]
        for col_value in table.col_values:
            key = (row_value, col_value)
            raw_value = _format_value(table.cells.get(key))
            raw_row.append(raw_value)
            if key in table.highlighted_cells:
                styled_row.append(_style_best_console(raw_value, highlight_styles))
            else:
                styled_row.append(raw_value)
        raw_rows.append(raw_row)
        styled_rows.append(styled_row)

    widths = _column_widths(columns, raw_rows)
    lines = [
        "Grid: "
        f"{_display_column(table.metric_name)} "
        f"(rows={_display_column(table.row_alias)}, "
        f"cols={_display_column(table.col_alias)})",
        _format_console_row(columns, widths),
        _format_console_row(["-" * width for width in widths], widths),
    ]
    for raw_row, styled_row in zip(raw_rows, styled_rows):
        lines.append(_format_console_row(styled_row, widths, raw_values=raw_row))
    return "\n".join(lines)


def render_flat_markdown(
    table: FlatTable,
    highlight_styles: Sequence[str] = (),
) -> str:
    """Render a flat table as Markdown."""
    display_columns = [_display_column(column) for column in table.columns]
    lines = [
        "| " + " | ".join(display_columns) + " |",
        "| " + " | ".join("---" for _ in table.columns) + " |",
    ]
    for row_idx, row in enumerate(table.rows):
        values = []
        for column in table.columns:
            value = _format_value(row.get(column))
            if row_idx in table.highlighted_columns.get(column, set()):
                value = _style_best_markdown(value, highlight_styles)
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def render_grid_markdown(
    table: GridTable,
    highlight_styles: Sequence[str] = (),
) -> str:
    """Render a pivot table as Markdown."""
    columns = [_display_column(table.row_alias), *[str(value) for value in table.col_values]]
    lines = [
        "### "
        f"{_display_column(table.metric_name)} "
        f"(rows={_display_column(table.row_alias)}, "
        f"cols={_display_column(table.col_alias)})",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row_value in table.row_values:
        values = [str(row_value)]
        for col_value in table.col_values:
            key = (row_value, col_value)
            value = _format_value(table.cells.get(key))
            if key in table.highlighted_cells:
                value = _style_best_markdown(value, highlight_styles)
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def render_flat_csv(table: FlatTable) -> str:
    """Render a flat table as CSV."""
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=list(table.columns), extrasaction="ignore")
    writer.writeheader()
    for row in table.rows:
        writer.writerow({column: row.get(column) for column in table.columns})
    return handle.getvalue()


def render_flat_latex(
    table: FlatTable,
    highlight_styles: Sequence[str] = (),
) -> str:
    """Render a flat table as a LaTeX table."""
    alignment = _latex_alignment_with_identifier_separator(table)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Experiment run summary}",
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\hline",
        _latex_row([_latex_escape(_display_column(column)) for column in table.columns]),
        "\\hline",
    ]
    for row_idx, row in enumerate(table.rows):
        values = []
        for column in table.columns:
            value = _latex_escape(_format_value(row.get(column)))
            if row_idx in table.highlighted_columns.get(column, set()):
                value = _style_best_latex(value, highlight_styles)
            values.append(value)
        lines.append(_latex_row(values))
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def render_flat_json(
    table: FlatTable,
    provenance: ReportingProvenance | None = None,
) -> str:
    """Render a flat table as JSON."""
    payload = {
        "columns": list(table.columns),
        "identifier_columns": list(table.identifier_columns),
        "rows": table.rows,
        "highlighted_columns": {
            column: sorted(indices)
            for column, indices in table.highlighted_columns.items()
        },
    }
    if provenance is not None:
        payload["provenance"] = provenance.to_dict()
    return json.dumps(payload, indent=2)


def render_grid_latex(
    table: GridTable,
    highlight_styles: Sequence[str] = (),
) -> str:
    """Render a pivot table as a LaTeX table."""
    columns = [_display_column(table.row_alias), *[str(value) for value in table.col_values]]
    alignment = "l" * len(columns)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{"
        + _latex_escape(
            f"{_display_column(table.metric_name)} by "
            f"{_display_column(table.row_alias)} and "
            f"{_display_column(table.col_alias)}"
        )
        + "}",
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\hline",
        _latex_row([_latex_escape(str(column)) for column in columns]),
        "\\hline",
    ]
    for row_value in table.row_values:
        values = [_latex_escape(str(row_value))]
        for col_value in table.col_values:
            key = (row_value, col_value)
            value = _latex_escape(_format_value(table.cells.get(key)))
            if key in table.highlighted_cells:
                value = _style_best_latex(value, highlight_styles)
            values.append(value)
        lines.append(_latex_row(values))
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def render_grid_json(
    tables: Sequence[GridTable],
    provenance: ReportingProvenance | None = None,
) -> str:
    """Render grid tables as JSON."""
    payload = []
    for table in tables:
        payload.append(
            {
                "metric_name": table.metric_name,
                "row_alias": table.row_alias,
                "col_alias": table.col_alias,
                "row_values": list(table.row_values),
                "col_values": list(table.col_values),
                "cells": [
                    {"row": row, "col": col, "value": value}
                    for (row, col), value in table.cells.items()
                ],
                "highlighted_cells": [
                    {"row": row, "col": col}
                    for row, col in sorted(
                        table.highlighted_cells,
                        key=lambda item: (str(item[0]), str(item[1])),
                    )
                ],
            }
        )
    result: Dict[str, Any] = {"grids": payload}
    if provenance is not None:
        result["provenance"] = provenance.to_dict()
    return json.dumps(result, indent=2)


def _column_widths(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> list[int]:
    widths = [len(str(column)) for column in columns]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))
    return widths


def _format_console_row(
    values: Sequence[str],
    widths: Sequence[int],
    raw_values: Sequence[str] | None = None,
) -> str:
    raw_values = raw_values or values
    padded = []
    for value, raw_value, width in zip(values, raw_values, widths):
        padded.append(str(value) + (" " * (width - len(str(raw_value)))))
    return "  ".join(padded)


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _display_column(column: str) -> str:
    return get_display_name(str(column))


def _with_provenance_comment(
    rendered_output: str,
    output_format: str,
    provenance: ReportingProvenance | None,
) -> str:
    if provenance is None:
        return rendered_output
    comment = _render_provenance_comment(provenance, output_format)
    if not comment:
        return rendered_output
    return f"{comment}\n{rendered_output}"


def _render_provenance_comment(
    provenance: ReportingProvenance,
    output_format: str,
) -> str:
    lines = provenance.to_lines()
    if output_format == "markdown":
        return "<!--\n" + "\n".join(lines) + "\n-->\n"
    if output_format == "latex":
        return "\n".join(f"% {line}" for line in lines) + "\n"
    if output_format in {"console", "csv"}:
        return "\n".join(f"# {line}" for line in lines) + "\n"
    return ""


def _style_best_console(value: str, highlight_styles: Sequence[str]) -> str:
    if value == "" or not highlight_styles:
        return value
    prefix = ""
    if "bold" in highlight_styles:
        prefix += ANSI_BOLD
    if "underline" in highlight_styles:
        prefix += ANSI_UNDERLINE
    return f"{prefix}{value}{ANSI_RESET}"


def _style_best_markdown(value: str, highlight_styles: Sequence[str]) -> str:
    if value == "" or not highlight_styles:
        return value
    if "bold" in highlight_styles:
        value = f"**{value}**"
    if "underline" in highlight_styles:
        value = f"<u>{value}</u>"
    return value


def _style_best_latex(value: str, highlight_styles: Sequence[str]) -> str:
    if value == "" or not highlight_styles:
        return value
    if "bold" in highlight_styles:
        value = f"\\textbf{{{value}}}"
    if "underline" in highlight_styles:
        value = f"\\underline{{{value}}}"
    return value


def _latex_row(values: Sequence[str]) -> str:
    return " & ".join(str(value) for value in values) + " \\\\"


def _latex_alignment_with_identifier_separator(table: FlatTable) -> str:
    identifier_count = len(table.identifier_columns)
    column_count = len(table.columns)
    if identifier_count <= 0 or identifier_count >= column_count:
        return "l" * column_count
    return ("l" * identifier_count) + "|" + ("l" * (column_count - identifier_count))


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)

