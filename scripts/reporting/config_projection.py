"""Project selected Hydra config paths into named reporting parameters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from scripts.reporting.run_reader import load_run_config
from scripts.reporting.schema import ParamAlias, RunSummary
from scripts.reporting.simple_yaml import select_path


def parse_param_alias(spec: str) -> ParamAlias:
    """Parse ``alias=config.path`` CLI syntax."""
    if "=" not in spec:
        raise ValueError(
            f"Invalid --param value '{spec}'. Expected syntax: alias=config.path."
        )
    alias, config_path = spec.split("=", 1)
    alias = alias.strip()
    config_path = config_path.strip()
    if not alias:
        raise ValueError(f"Invalid --param value '{spec}': alias is empty.")
    if not config_path:
        raise ValueError(f"Invalid --param value '{spec}': config path is empty.")
    return ParamAlias(alias=alias, config_path=config_path)


def parse_param_aliases(specs: Iterable[str]) -> Sequence[ParamAlias]:
    """Parse repeated ``--param`` arguments."""
    aliases = [parse_param_alias(spec) for spec in specs]
    seen: set[str] = set()
    for alias in aliases:
        if alias.alias in seen:
            raise ValueError(f"Duplicate parameter alias '{alias.alias}'.")
        seen.add(alias.alias)
    return aliases


def project_params(cfg: Dict[str, Any], aliases: Sequence[ParamAlias]) -> Dict[str, Any]:
    """Resolve configured aliases from a loaded Hydra config."""
    values: Dict[str, Any] = {}
    missing = object()
    for alias in aliases:
        value = select_path(cfg, alias.config_path, default=missing)
        if value is missing:
            raise KeyError(
                f"Config path '{alias.config_path}' for alias '{alias.alias}' "
                "was not found."
            )
        values[alias.alias] = _normalize_report_value(value, alias)
    return values


def attach_projected_params(
    summaries: Sequence[RunSummary],
    aliases: Sequence[ParamAlias],
) -> None:
    """Load each run config and attach projected parameter values in-place."""
    if not aliases:
        return

    for summary in summaries:
        if summary.config_path is None:
            summary.status = "missing_config"
            summary.errors.append(f"Missing config path for run {summary.run_name}")
            continue
        try:
            cfg = load_run_config(Path(summary.run_dir))
            summary.params = project_params(cfg, aliases)
        except (OSError, KeyError, ValueError, TypeError) as exc:
            if summary.status == "complete":
                summary.status = "invalid_config_projection"
            summary.errors.append(str(exc))


def _normalize_report_value(value: Any, alias: ParamAlias) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        if not all(_is_report_scalar(item) for item in value):
            raise TypeError(
                f"Config path '{alias.config_path}' for alias '{alias.alias}' "
                "resolved to a nested sequence or mapping."
            )
        return "[" + ", ".join(str(item) for item in value) + "]"
    if isinstance(value, tuple):
        if not all(_is_report_scalar(item) for item in value):
            raise TypeError(
                f"Config path '{alias.config_path}' for alias '{alias.alias}' "
                "resolved to a nested sequence or mapping."
            )
        return "[" + ", ".join(str(item) for item in value) + "]"
    if isinstance(value, dict):
        raise TypeError(
            f"Config path '{alias.config_path}' for alias '{alias.alias}' "
            "resolved to a mapping."
        )
    raise TypeError(
        f"Config path '{alias.config_path}' for alias '{alias.alias}' "
        f"resolved to unsupported value type {type(value).__name__}."
    )


def _is_report_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))

