"""
Config-driven repository-model evaluation pipeline.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from omegaconf import DictConfig, OmegaConf

from scripts.evaluation.core.contracts import EvaluationThresholdProtocol, VolumeSample
from scripts.evaluation.io.model_volumes import (
    iter_model_volume_samples,
    validate_model_evaluation_mode,
)
from scripts.evaluation.metrics.registry_3d import (
    THREED_METRIC_CLASSES,
    compute_metrics_3d_at_threshold,
    resolve_3d_metric_class_names,
)
from scripts.evaluation.core.model_config import (
    resolve_evaluation_output_dir,
    write_resolved_evaluation_config,
)
from scripts.evaluation.core.model_loader import (
    build_model_for_evaluation,
    find_checkpoint,
    resolve_diffusion_type,
)
from scripts.evaluation.reporting import write_json_report
from scripts.evaluation.reporting.threshold_protocol import build_evaluation_threshold_protocol
from scripts.evaluation.reporting.threshold_records import (
    ThresholdMetricRecord,
    add_volume_ratio,
    aggregate_threshold_records,
    select_global_threshold,
    select_oracle_thresholds,
    write_oracle_threshold_csv,
    write_per_case_threshold_csv,
)
from src.data.loaders import get_dataloaders


@dataclass(frozen=True)
class ModelEvaluationRequest:
    """Typed request resolved from the final evaluation config."""

    run_dir: Path
    model_name: str
    checkpoint_path: Path
    output_dir: Path
    device: str
    levels: Sequence[str]
    threshold_protocol: EvaluationThresholdProtocol
    use_ema: bool
    loader_mode: str
    data_dim: str
    diffusion_type: str


def build_model_evaluation_request(cfg: DictConfig) -> ModelEvaluationRequest:
    """
    Resolve and validate a model evaluation request from config.
    """
    input_source = str(OmegaConf.select(cfg, "evaluation.input_source", default="live_model"))
    if input_source != "live_model":
        raise ValueError(
            "Only evaluation.input_source='live_model' is supported in this pipeline. "
            f"Got {input_source!r}."
        )

    run_dir_value = OmegaConf.select(cfg, "evaluation.run_dir", default=None)
    if not _is_set(run_dir_value):
        raise ValueError("evaluation.run_dir is required.")
    run_dir = Path(str(run_dir_value))

    model_name_value = OmegaConf.select(cfg, "evaluation.model_name", default=None)
    if not _is_set(model_name_value):
        raise ValueError("evaluation.model_name is required.")
    model_name = str(model_name_value)

    levels = _resolve_levels(cfg)
    use_ema = bool(OmegaConf.select(cfg, "evaluation.checkpoint.use_ema", default=False))
    checkpoint_path = find_checkpoint(
        run_dir=run_dir,
        model_name=model_name,
        use_ema=use_ema,
    )
    output_dir = resolve_evaluation_output_dir(cfg)
    device = _resolve_device(cfg)
    threshold_protocol = build_evaluation_threshold_protocol(cfg)
    validate_model_evaluation_mode(cfg)

    return ModelEvaluationRequest(
        run_dir=run_dir,
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        device=device,
        levels=levels,
        threshold_protocol=threshold_protocol,
        use_ema=use_ema,
        loader_mode=str(OmegaConf.select(cfg, "data_mode.loader_mode", default="") or ""),
        data_dim=_normalize_dim_token(OmegaConf.select(cfg, "data_mode.dim", default=None)),
        diffusion_type=resolve_diffusion_type(cfg),
    )


def run_model_evaluation(cfg: DictConfig) -> Dict[str, Any]:
    """
    Execute repository-model evaluation and write artifacts.
    """
    request = build_model_evaluation_request(cfg)
    request.output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model_for_evaluation(
        cfg=cfg,
        checkpoint_path=request.checkpoint_path,
        device=request.device,
    )
    dataloaders = get_dataloaders(cfg)
    if "val" not in dataloaders:
        raise ValueError("get_dataloaders(cfg) did not return a 'val' dataloader.")

    if request.data_dim == "3d" and "volume" in request.levels:
        evaluation_result = evaluate_model_volumes(
            model=model,
            dataloader=dataloaders["val"],
            cfg=cfg,
            request=request,
        )
    elif request.data_dim == "2d":
        raise NotImplementedError(
            "2D model evaluation pipeline migration is deferred. "
            "Use the existing 2D evaluation entrypoint for now."
        )
    else:
        raise ValueError(
            f"Unsupported evaluation request: data_dim={request.data_dim}, "
            f"levels={list(request.levels)}."
        )

    return _write_model_evaluation_artifacts(
        cfg=cfg,
        request=request,
        evaluation_result=evaluation_result,
    )


def evaluate_model_volumes(
    model: Any,
    dataloader: Iterable[Any],
    cfg: DictConfig,
    request: ModelEvaluationRequest,
) -> Dict[str, Any]:
    """
    Evaluate live model volume predictions across configured thresholds.
    """
    records: List[ThresholdMetricRecord] = []
    metric_names = _resolve_volume_metric_names(cfg)
    sample_count = 0

    for sample in iter_model_volume_samples(
        model=model,
        dataloader=dataloader,
        cfg=cfg,
        device=request.device,
        show_progress=bool(OmegaConf.select(cfg, "evaluation.show_progress", default=True)),
    ):
        sample_count += 1
        sample_records = _evaluate_volume_sample(
            sample=sample,
            thresholds=request.threshold_protocol.thresholds,
            metric_names=metric_names,
        )
        records.extend(sample_records)

    aggregates = aggregate_threshold_records(records, selector_level="volume")
    if not aggregates:
        raise RuntimeError("Volume evaluation produced no threshold records.")

    global_selection = None
    if request.threshold_protocol.mode in {"sweep", "sweep_with_oracle"}:
        global_selection = select_global_threshold(
            records=records,
            selector=request.threshold_protocol.primary,
        )

    oracle_rows: Optional[List[Dict[str, object]]] = None
    oracle_summary: Optional[Dict[str, object]] = None
    if request.threshold_protocol.mode in {"oracle_per_case", "sweep_with_oracle"}:
        oracle_rows, oracle_summary = select_oracle_thresholds(
            records=records,
            selector=request.threshold_protocol.primary,
        )

    return {
        "records": records,
        "aggregates": aggregates,
        "global_selection": global_selection,
        "oracle_rows": oracle_rows,
        "oracle_summary": oracle_summary,
        "sample_count": int(sample_count),
        "metric_names": list(metric_names),
    }


def _evaluate_volume_sample(
    sample: VolumeSample,
    thresholds: Sequence[float],
    metric_names: Sequence[str],
) -> List[ThresholdMetricRecord]:
    sample.validate()
    metric_configs = _build_spacing_metric_configs(sample.metadata)
    records: List[ThresholdMetricRecord] = []
    for threshold in thresholds:
        metrics = compute_metrics_3d_at_threshold(
            pred=sample.prediction_volume,
            gt=sample.ground_truth_volume,
            threshold=float(threshold),
            metric_configs=metric_configs,
            metric_names=metric_names,
        )
        metrics = add_volume_ratio(metrics)
        records.append(
            ThresholdMetricRecord(
                level="volume",
                case_id=str(sample.case_id),
                threshold=float(threshold),
                metrics=metrics,
                metadata={
                    "volume_id": str(sample.volume_id),
                    **dict(sample.metadata),
                },
            )
        )
    return records


def _write_model_evaluation_artifacts(
    cfg: DictConfig,
    request: ModelEvaluationRequest,
    evaluation_result: Mapping[str, Any],
) -> Dict[str, Any]:
    records = list(evaluation_result["records"])
    aggregates = dict(evaluation_result["aggregates"])
    oracle_rows = evaluation_result.get("oracle_rows")

    payload = build_model_evaluation_payload(
        request=request,
        evaluation_result=evaluation_result,
    )
    json_path = write_json_report(payload, output_dir=request.output_dir)
    summary_text = build_model_evaluation_summary(payload)
    summary_path = request.output_dir / "evaluation_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")

    aggregate_csv_path = write_aggregate_threshold_csv(
        aggregates=aggregates,
        output_dir=request.output_dir,
        filename="volume_metrics_per_threshold.csv",
    )
    per_case_csv_path = write_per_case_threshold_csv(
        records=records,
        output_dir=request.output_dir,
    )

    oracle_csv_path = None
    if oracle_rows is not None:
        oracle_csv_path = write_oracle_threshold_csv(
            rows=oracle_rows,
            output_dir=request.output_dir,
        )

    config_path = None
    if bool(OmegaConf.select(cfg, "evaluation.reporting.write_config", default=True)):
        config_path = write_resolved_evaluation_config(cfg, request.output_dir)

    paths = {
        "json_path": str(json_path),
        "summary_path": str(summary_path),
        "volume_csv_path": str(aggregate_csv_path),
        "per_case_csv_path": str(per_case_csv_path),
        "oracle_csv_path": str(oracle_csv_path) if oracle_csv_path is not None else None,
        "config_path": str(config_path) if config_path is not None else None,
    }
    return {
        "output_dir": str(request.output_dir),
        "paths": paths,
        "summary_text": summary_text,
        "selected_global_threshold": (
            payload["threshold_analysis"]["best_global_threshold"]["threshold"]
            if payload["threshold_analysis"].get("best_global_threshold") is not None
            else None
        ),
        "oracle_summary": payload["threshold_analysis"].get("oracle_per_case"),
        **paths,
    }


def build_model_evaluation_payload(
    request: ModelEvaluationRequest,
    evaluation_result: Mapping[str, Any],
) -> Dict[str, object]:
    """
    Build the canonical JSON payload for repository-model evaluation.
    """
    aggregates = dict(evaluation_result["aggregates"])
    ordered_thresholds = sorted(float(threshold) for threshold in aggregates.keys())
    threshold_rows = [aggregates[threshold] for threshold in ordered_thresholds]
    fixed_threshold_row = _lookup_threshold_row(
        aggregates=aggregates,
        threshold=request.threshold_protocol.fixed_threshold,
    )

    global_selection = evaluation_result.get("global_selection")
    oracle_summary = evaluation_result.get("oracle_summary")
    payload = {
        "metadata": {
            "entrypoint": "evaluate_model",
            "run_dir": str(request.run_dir),
            "model_name": request.model_name,
            "checkpoint_path": str(request.checkpoint_path),
            "output_dir": str(request.output_dir),
            "device": request.device,
            "use_ema": bool(request.use_ema),
        },
        "data_summary": {
            "levels": list(request.levels),
            "data_dim": request.data_dim,
            "loader_mode": request.loader_mode,
            "diffusion_type": request.diffusion_type,
            "total_volumes": int(evaluation_result.get("sample_count", 0)),
        },
        "protocol": {
            "mode": request.threshold_protocol.mode,
            "thresholds_evaluated": [float(t) for t in request.threshold_protocol.thresholds],
            "fixed_threshold": float(request.threshold_protocol.fixed_threshold),
            "primary_selector": _selector_to_dict(request.threshold_protocol.primary),
        },
        "metrics": {
            "volume_level": {
                "metric_names": list(evaluation_result.get("metric_names", [])),
                "threshold_results": threshold_rows,
            }
        },
        "threshold_analysis": {
            "fixed_threshold": fixed_threshold_row,
            "best_global_threshold": global_selection,
            "oracle_per_case": oracle_summary,
            "primary_selector": _selector_to_dict(request.threshold_protocol.primary),
        },
    }
    return payload


def build_model_evaluation_summary(payload: Mapping[str, Any]) -> str:
    """
    Build a concise text summary from the model-evaluation payload.
    """
    metadata = payload["metadata"]
    data_summary = payload["data_summary"]
    protocol = payload["protocol"]
    analysis = payload["threshold_analysis"]
    lines = [
        "Repository Model Evaluation Summary",
        "=" * 50,
        f"Run dir:     {metadata['run_dir']}",
        f"Model:       {metadata['model_name']}",
        f"Checkpoint:  {metadata['checkpoint_path']}",
        f"Output dir:  {metadata['output_dir']}",
        "",
        "Data:",
        f"  Dim:        {data_summary['data_dim']}",
        f"  Loader:     {data_summary['loader_mode']}",
        f"  Volumes:    {data_summary['total_volumes']}",
        "",
        "Protocol:",
        f"  Mode:              {protocol['mode']}",
        f"  Thresholds:        {protocol['thresholds_evaluated']}",
        f"  Fixed threshold:   {protocol['fixed_threshold']}",
        "",
    ]

    fixed_row = analysis.get("fixed_threshold")
    if fixed_row is not None:
        lines.extend(_format_threshold_block("Fixed Threshold", fixed_row, protocol["primary_selector"]))

    best_global = analysis.get("best_global_threshold")
    if best_global is not None:
        lines.extend(
            [
                "Best Global Threshold:",
                f"  Threshold: {best_global['threshold']}",
                f"  Selected value: {best_global['selected_statistic_value']:.6f}",
                "",
            ]
        )

    oracle = analysis.get("oracle_per_case")
    if oracle is not None:
        lines.extend(
            [
                "Oracle Per Case:",
                f"  Cases: {oracle['case_count']}",
                f"  Threshold counts: {oracle['threshold_counts']}",
                "",
            ]
        )

    lines.append("=" * 50)
    return "\n".join(lines)


def write_aggregate_threshold_csv(
    aggregates: Mapping[float, Mapping[str, object]],
    output_dir: Path,
    filename: str = "volume_metrics_per_threshold.csv",
) -> Path:
    """
    Write aggregate per-threshold metrics, including median/min/max statistics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    metric_names = sorted(
        {
            str(metric_name)
            for row in aggregates.values()
            for metric_name in _mapping_or_empty(row.get("metrics")).keys()
        }
    )
    stats = ("count", "mean", "median", "std", "min", "max")
    fieldnames = ["level", "threshold", "case_count", "record_count"]
    for metric_name in metric_names:
        fieldnames.extend(f"{metric_name}_{stat}" for stat in stats)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for threshold in sorted(float(t) for t in aggregates.keys()):
            source = aggregates[threshold]
            row = {
                "level": source.get("level", "volume"),
                "threshold": float(threshold),
                "case_count": int(source.get("case_count", 0)),
                "record_count": int(source.get("record_count", 0)),
            }
            metrics = _mapping_or_empty(source.get("metrics"))
            for metric_name in metric_names:
                metric_stats = _mapping_or_empty(metrics.get(metric_name))
                for stat in stats:
                    row[f"{metric_name}_{stat}"] = metric_stats.get(stat, "")
            writer.writerow(row)
    return path


def _resolve_levels(cfg: DictConfig) -> List[str]:
    raw_levels = OmegaConf.select(cfg, "evaluation.levels", default=["volume"])
    levels = [str(level).strip().lower() for level in list(raw_levels)]
    if not levels:
        raise ValueError("evaluation.levels must not be empty.")
    invalid = [level for level in levels if level not in {"slice", "volume"}]
    if invalid:
        raise ValueError(f"Unsupported evaluation levels: {invalid}.")
    return levels


def _resolve_device(cfg: DictConfig) -> str:
    configured = OmegaConf.select(cfg, "evaluation.device", default=None)
    if _is_set(configured):
        return str(configured)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_volume_metric_names(cfg: DictConfig) -> Sequence[str]:
    configured = OmegaConf.select(cfg, "evaluation.metrics_3d.names", default=None)
    if configured is not None:
        metric_names = [str(name) for name in list(configured)]
        if not metric_names:
            raise ValueError("evaluation.metrics_3d.names must not be empty when provided.")
        unknown = [name for name in metric_names if name not in THREED_METRIC_CLASSES]
        if unknown:
            raise ValueError(
                "evaluation.metrics_3d.names must use 3D metric class names, "
                f"not validation aliases. Unknown class-name keys: {unknown}. "
                f"Available class names: {sorted(THREED_METRIC_CLASSES)}"
            )
        return tuple(metric_names)

    validation_metric_names = _resolve_validation_3d_metric_aliases(cfg)
    if validation_metric_names is not None:
        return resolve_3d_metric_class_names(validation_metric_names)

    return tuple(THREED_METRIC_CLASSES.keys())


def _resolve_validation_3d_metric_aliases(cfg: DictConfig) -> Optional[Sequence[str]]:
    metric_configs = OmegaConf.select(cfg, "validation.metrics", default=None)
    if metric_configs is None:
        return None
    for metric_config in metric_configs:
        name = str(metric_config.get("name", "")).strip()
        if name != "ThreeDMetricsAggregator":
            continue
        enabled_metrics = metric_config.get("params", {}).get("enabled_metrics", None)
        if enabled_metrics is None:
            return None
        if isinstance(enabled_metrics, str):
            metric_names = [enabled_metrics]
        else:
            metric_names = [str(metric_name) for metric_name in list(enabled_metrics)]
        if not metric_names:
            raise ValueError(
                "validation.metrics ThreeDMetricsAggregator params.enabled_metrics "
                "must not be empty when provided."
            )
        return metric_names
    return None


def _build_spacing_metric_configs(metadata: Mapping[str, object]) -> Dict[str, Dict[str, object]]:
    spacing = metadata.get("source_spacing_xyz")
    if spacing is None:
        spacing_xyz = (1.0, 1.0, 1.0)
    else:
        spacing_values = tuple(float(value) for value in list(spacing)[:3])
        spacing_xyz = spacing_values if len(spacing_values) == 3 else (1.0, 1.0, 1.0)
    voxel_size = float(spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2])
    return {
        "AbsoluteVolumeDifferenceNative": {"voxel_size": voxel_size},
        "HausdorffDistance95MonaiMm": {"spacing": spacing_xyz},
        "SurfaceDiceMonai": {"spacing": spacing_xyz, "tolerance_mm": 1.0},
        "PredictedVolumeMm3": {"spacing": spacing_xyz},
        "GroundTruthVolumeMm3": {"spacing": spacing_xyz},
    }


def _lookup_threshold_row(
    aggregates: Mapping[float, Mapping[str, object]],
    threshold: float,
    tol: float = 1e-9,
) -> Optional[Mapping[str, object]]:
    for key, row in aggregates.items():
        if abs(float(key) - float(threshold)) <= tol:
            return row
    return None


def _format_threshold_block(
    title: str,
    row: Mapping[str, object],
    selector: Mapping[str, str],
) -> List[str]:
    metric_name = selector["metric"]
    statistic = selector["statistic"]
    metrics = _mapping_or_empty(row.get("metrics"))
    metric_stats = _mapping_or_empty(metrics.get(metric_name))
    value = metric_stats.get(statistic, None)
    value_text = "n/a" if value is None else f"{float(value):.6f}"
    return [
        f"{title}:",
        f"  Threshold: {row.get('threshold')}",
        f"  {metric_name} {statistic}: {value_text}",
        "",
    ]


def _selector_to_dict(selector: Any) -> Dict[str, str]:
    return {
        "level": str(selector.level),
        "metric": str(selector.metric),
        "statistic": str(selector.statistic),
        "direction": str(selector.direction),
    }


def _mapping_or_empty(value: object) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    return {}


def _normalize_dim_token(value: object) -> str:
    if value is None:
        raise ValueError("Missing data_mode.dim for model evaluation.")
    token = str(value).strip().lower()
    if token in {"2", "2d"}:
        return "2d"
    if token in {"3", "3d"}:
        return "3d"
    return token


def _is_set(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True
