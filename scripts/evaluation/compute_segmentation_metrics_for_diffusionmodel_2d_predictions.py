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

from scripts.analysis.threshold_analysis import (
    find_checkpoint,
    load_config_from_run_dir,
    load_model,
)
from scripts.evaluation.io_diffusion import iter_diffusion_case_slice_samples
from scripts.evaluation.metrics_engine import StreamingMetricsEngine
from scripts.evaluation.reporting import (
    build_report_payload,
    build_text_summary,
    write_json_report,
    write_threshold_csv,
)
from scripts.evaluation.threshold_protocol import (
    make_fixed_protocol,
    make_sweep_protocol_from_spec,
    select_primary_threshold,
)
from src.data.loaders import get_dataloaders


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
    checkpoint_path = find_checkpoint(str(args.run_dir), args.model_name, use_ema=args.ema)
    model = load_model(cfg, checkpoint_path, device)
    dataloaders = get_dataloaders(cfg)
    val_loader = dataloaders["val"]

    analysis_cases = _build_analysis_cases(args.ensemble_samples_list, args.ensemble_method)
    use_filename_prefix = len(analysis_cases) > 1
    engines = {case["key"]: StreamingMetricsEngine(thresholds=protocol.thresholds) for case in analysis_cases}

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
        engines[case_key].update(sample)

    print("\nOutputs:")
    for case in analysis_cases:
        case_key = str(case["key"])
        case_label = str(case["label"])
        filename_prefix = f"{case_key}_" if use_filename_prefix else ""
        finalized_results = engines[case_key].finalize()
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
        summary_path = output_dir / f"{filename_prefix}evaluation_summary.txt"
        summary_text = build_text_summary(payload)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary_text, encoding="utf-8")

        print(f"  [{case_label}]")
        print(f"    {json_path}")
        print(f"    {csv_path}")
        print(f"    {summary_path}")


if __name__ == "__main__":
    main()

