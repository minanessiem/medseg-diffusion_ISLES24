#!/usr/bin/env python3
"""
Compute canonical segmentation metrics for custom diffusion/discriminative 2D models.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
from omegaconf import DictConfig, OmegaConf

from scripts.analysis.threshold_analysis import (
    find_checkpoint,
    load_config_from_run_dir,
    load_model,
)
from scripts.evaluation.io_diffusion import iter_diffusion_case_slice_samples
from scripts.evaluation.metrics_engine import DualLevelStreamingMetricsEngine
from scripts.evaluation.reporting import (
    build_report_payload,
    build_text_summary,
    write_json_report,
    write_threshold_csv,
    write_volume_threshold_csv,
)
from scripts.evaluation.threshold_protocol import (
    make_fixed_protocol,
    make_sweep_protocol_from_spec,
    select_primary_threshold,
)
from scripts.evaluation.volume_exporter import export_reconstructed_volumes
from src.data.loaders import get_dataloaders, validate_dataset_contract


def _is_set(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return len(value.strip()) > 0
    return True


def validate_eval_config_contract(cfg: DictConfig) -> None:
    """
    Validate config preconditions for 2D diffusion/custom evaluation.
    """
    validate_dataset_contract(cfg)

    loader_mode = OmegaConf.select(cfg, "data_mode.loader_mode")
    dim = OmegaConf.select(cfg, "data_mode.dim")
    supported_loader_modes = {"online_slices_3d_to_2d", "nnunet_slices_2d"}

    if dim != "2d":
        raise ValueError(
            "Diffusion/custom 2D evaluation requires data_mode.dim='2d'. "
            f"Got '{dim}'."
        )
    if loader_mode not in supported_loader_modes:
        allowed = ", ".join(sorted(supported_loader_modes))
        raise ValueError(
            "Diffusion/custom 2D evaluation requires a slice-based loader_mode. "
            f"Expected one of [{allowed}], got '{loader_mode}'."
        )
    if not _is_set(OmegaConf.select(cfg, "validation.val_batch_size")):
        raise ValueError(
            "Missing required key for evaluation: validation.val_batch_size."
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute canonical segmentation metrics for custom model 2D predictions",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory containing checkpoints and .hydra config",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model checkpoint name without .pth extension",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <run-dir>/analysis/evaluation_v2/<model>_<timestamp>)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--ema",
        action="store_true",
        help="Use EMA checkpoint if available",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.05:0.95:0.05",
        help='Sweep thresholds as "start:stop:step" or comma-separated list',
    )
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=None,
        help="If provided, skip sweep and evaluate only this threshold",
    )
    parser.add_argument(
        "--optimize-metric",
        type=str,
        default="dice",
        help="Metric for selecting primary threshold in sweep mode",
    )
    parser.add_argument(
        "--ensemble-samples",
        type=str,
        default="1",
        help="Single integer or comma-separated list, e.g. '1' or '1,3,5'",
    )
    parser.add_argument(
        "--ensemble-method",
        type=str,
        default="single",
        choices=["single", "mean", "soft_staple", "both"],
        help="Prediction merge method (single, mean, soft_staple, or both)",
    )
    parser.add_argument(
        "--staple-max-iters",
        type=int,
        default=5,
        help="Maximum STAPLE iterations (default: 5)",
    )
    parser.add_argument(
        "--staple-tolerance",
        type=float,
        default=0.02,
        help="STAPLE convergence tolerance (default: 0.02)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test mode: process only a small number of slices",
    )
    parser.add_argument(
        "--test-max-slices",
        type=int,
        default=10,
        help="Number of slices to process in --test mode (default: 10)",
    )
    parser.add_argument(
        "--export-reconstructed-volumes",
        action="store_true",
        help="Export reconstructed prediction/GT 3D volumes as NIfTI.",
    )
    parser.add_argument(
        "--max-export-volumes-per-case",
        type=int,
        default=None,
        help="Optional cap for exported reconstructed volumes per analysis case.",
    )
    return parser


def _parse_ensemble_samples(spec: str) -> List[int]:
    parsed: List[int] = []
    for part in str(spec).split(","):
        value = part.strip()
        if not value:
            continue
        sample_count = int(value)
        if sample_count <= 0:
            raise ValueError(f"Ensemble sample counts must be > 0, got {sample_count}")
        parsed.append(sample_count)
    if not parsed:
        raise ValueError("No valid ensemble sample counts provided.")
    return list(dict.fromkeys(parsed))


def _build_analysis_cases(ensemble_samples_list: List[int], ensemble_method: str) -> List[Dict[str, object]]:
    cases: List[Dict[str, object]] = []
    for sample_count in ensemble_samples_list:
        if sample_count == 1:
            cases.append(
                {
                    "key": "n1_single",
                    "label": "n=1 (first sample)",
                    "method": "single",
                    "num_samples": 1,
                }
            )
            continue
        methods = ["mean", "soft_staple"] if ensemble_method == "both" else [ensemble_method]
        for method in methods:
            if method == "single":
                continue
            cases.append(
                {
                    "key": f"n{sample_count}_{method}",
                    "label": f"n={sample_count}, method={method}",
                    "method": method,
                    "num_samples": sample_count,
                }
            )
    if not cases:
        raise ValueError("No analysis cases were generated from ensemble settings.")
    return cases


def validate_args(args: argparse.Namespace) -> None:
    try:
        args.ensemble_samples_list = _parse_ensemble_samples(args.ensemble_samples)
    except ValueError as exc:
        raise ValueError(f"Invalid --ensemble-samples: {exc}") from exc
    args.max_ensemble_samples = max(args.ensemble_samples_list)

    if args.ensemble_method == "single" and args.max_ensemble_samples > 1:
        raise ValueError("--ensemble-method=single cannot be used with ensemble sizes > 1.")
    if args.ensemble_method in {"mean", "soft_staple"} and args.max_ensemble_samples < 2:
        raise ValueError(
            f"--ensemble-method={args.ensemble_method} requires at least one ensemble size >= 2."
        )
    if args.ensemble_method == "both" and args.max_ensemble_samples < 2:
        raise ValueError("--ensemble-method=both requires at least one ensemble size >= 2.")
    if args.test_max_slices <= 0:
        raise ValueError("--test-max-slices must be > 0.")


def main() -> None:
    args = build_arg_parser().parse_args()
    validate_args(args)

    if not args.run_dir.exists():
        print(f"Error: run directory does not exist: {args.run_dir}")
        sys.exit(1)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.fixed_threshold is not None:
        protocol = make_fixed_protocol(args.fixed_threshold)
    else:
        protocol = make_sweep_protocol_from_spec(
            threshold_spec=args.thresholds,
            optimize_metric=args.optimize_metric,
        )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = args.output_dir or (
        args.run_dir / "analysis" / "evaluation_v2" / f"{args.model_name}_{timestamp}"
    )

    print("=" * 60)
    print("EVALUATION V2 (custom diffusion/discriminative model)")
    print("=" * 60)
    print(f"Run dir: {args.run_dir}")
    print(f"Model: {args.model_name}")
    print(f"EMA: {args.ema}")
    print(f"Device: {device}")
    print(f"Protocol: {protocol.mode}")
    print(f"Thresholds: {protocol.thresholds}")
    print(f"Optimize metric: {protocol.optimize_metric}")
    requested_sizes_str = ",".join(str(n) for n in args.ensemble_samples_list)
    print(
        f"Ensemble: requested_sizes=[{requested_sizes_str}], "
        f"max_samples={args.max_ensemble_samples}, method={args.ensemble_method}"
    )
    if args.test:
        print(f"Test mode: enabled (max_slices={args.test_max_slices})")
    print("=" * 60)

    cfg = load_config_from_run_dir(str(args.run_dir))
    validate_eval_config_contract(cfg)
    checkpoint_path = find_checkpoint(str(args.run_dir), args.model_name, use_ema=args.ema)
    model = load_model(cfg, checkpoint_path, device)
    dataloaders = get_dataloaders(cfg)
    val_loader = dataloaders["val"]
    if hasattr(val_loader, "dataset") and hasattr(val_loader.dataset, "return_metadata"):
        val_loader.dataset.return_metadata = True

    analysis_cases = _build_analysis_cases(args.ensemble_samples_list, args.ensemble_method)
    use_filename_prefix = len(analysis_cases) > 1
    engines = {
        case["key"]: DualLevelStreamingMetricsEngine(
            thresholds=protocol.thresholds,
            assembler_case_key=str(case["key"]),
        )
        for case in analysis_cases
    }
    export_dir = output_dir / "reconstructed_volumes" if args.export_reconstructed_volumes else None
    exported_volume_counts = {str(case["key"]): 0 for case in analysis_cases}

    case_samples = iter_diffusion_case_slice_samples(
        model=model,
        dataloader=val_loader,
        device=device,
        analysis_cases=analysis_cases,
        max_requested_size=args.max_ensemble_samples,
        staple_max_iters=args.staple_max_iters,
        staple_tolerance=args.staple_tolerance,
        max_samples=args.test_max_slices if args.test else None,
    )
    for case_key, sample in case_samples:
        finalized_volumes = engines[case_key].update(sample)
        if args.export_reconstructed_volumes and finalized_volumes:
            remaining = _remaining_export_budget(
                exported=exported_volume_counts[case_key],
                max_per_case=args.max_export_volumes_per_case,
            )
            if remaining != 0:
                to_export = finalized_volumes if remaining < 0 else finalized_volumes[:remaining]
                export_reconstructed_volumes(
                    grouped_volumes={case_key: to_export},
                    output_dir=export_dir,
                    max_volumes_per_case=None,
                )
                exported_volume_counts[case_key] += len(to_export)

    print("\nOutputs:")
    for case in analysis_cases:
        case_key = str(case["key"])
        case_label = str(case["label"])
        filename_prefix = f"{case_key}_" if use_filename_prefix else ""
        trailing_volumes = engines[case_key].finalize_open_volumes()
        if args.export_reconstructed_volumes and trailing_volumes:
            remaining = _remaining_export_budget(
                exported=exported_volume_counts[case_key],
                max_per_case=args.max_export_volumes_per_case,
            )
            if remaining != 0:
                to_export = trailing_volumes if remaining < 0 else trailing_volumes[:remaining]
                export_reconstructed_volumes(
                    grouped_volumes={case_key: to_export},
                    output_dir=export_dir,
                    max_volumes_per_case=None,
                )
                exported_volume_counts[case_key] += len(to_export)

        dual_level_results = engines[case_key].finalize()
        finalized_results = dual_level_results["slice_level"]
        volume_results = dual_level_results["volume_level"]
        selected_threshold = select_primary_threshold(finalized_results, protocol)

        payload = build_report_payload(
            finalized_results=finalized_results,
            protocol=protocol,
            entrypoint_name="compute_segmentation_metrics_for_diffusionmodel_2d_predictions",
            metadata={
                "run_dir": str(args.run_dir.resolve()),
                "model_name": args.model_name,
                "checkpoint_path": checkpoint_path,
                "use_ema": bool(args.ema),
                "requested_ensemble_sizes": args.ensemble_samples_list,
                "ensemble_method": args.ensemble_method,
                "analysis_case_key": case_key,
                "analysis_case_label": case_label,
                "analysis_case_num_samples": int(case["num_samples"]),
                "analysis_case_method": str(case["method"]),
                "device": device,
                "test_mode": bool(args.test),
                "test_max_slices": int(args.test_max_slices) if args.test else None,
            },
            selected_threshold=selected_threshold,
            volume_finalized_results=volume_results,
        )

        json_path = write_json_report(
            payload,
            output_dir=output_dir,
            filename=f"{filename_prefix}canonical_results.json",
        )
        csv_path = write_threshold_csv(
            finalized_results,
            output_dir=output_dir,
            filename=f"{filename_prefix}metrics_per_threshold.csv",
        )
        volume_csv_path = write_volume_threshold_csv(
            volume_results,
            output_dir=output_dir,
            filename=f"{filename_prefix}volume_metrics_per_threshold.csv",
        )
        summary_path = output_dir / f"{filename_prefix}evaluation_summary.txt"
        summary_text = build_text_summary(payload)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary_text, encoding="utf-8")

        print(f"  [{case_label}]")
        print(f"    {json_path}")
        print(f"    {csv_path}")
        print(f"    {volume_csv_path}")
        print(f"    {summary_path}")
        if args.export_reconstructed_volumes:
            print(
                f"    exported_reconstructed_volumes={exported_volume_counts[case_key]} "
                f"-> {export_dir / case_key}"
            )


def _remaining_export_budget(exported: int, max_per_case: int | None) -> int:
    """
    Remaining export slots for one case.

    Returns:
        -1 when unlimited.
        0 when cap reached.
        >0 when that many slots remain.
    """
    if max_per_case is None:
        return -1
    remaining = int(max_per_case) - int(exported)
    return max(remaining, 0)


if __name__ == "__main__":
    main()

