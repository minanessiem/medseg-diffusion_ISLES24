#!/usr/bin/env python3
"""
Compute canonical segmentation metrics for nnU-Net 2D post-threshold predictions.
"""

import argparse
import sys
from pathlib import Path

from scripts.evaluation.io_nnunet import count_matched_pairs, iter_nnunet_slice_samples
from scripts.evaluation.metrics_engine import DualLevelStreamingMetricsEngine
from scripts.evaluation.reporting import (
    build_report_payload,
    build_text_summary,
    write_json_report,
    write_threshold_csv,
    write_volume_threshold_csv,
)
from scripts.evaluation.threshold_protocol import (
    enforce_post_threshold_mode,
    make_fixed_protocol,
    select_primary_threshold,
)
from scripts.evaluation.volume_exporter import export_reconstructed_volumes


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute canonical segmentation metrics for nnU-Net 2D predictions",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        required=True,
        help="Directory containing nnU-Net prediction NIfTI files (*.nii.gz)",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        required=True,
        help="Directory containing ground truth NIfTI files (*.nii.gz)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <pred-dir>/evaluation_v2)",
    )
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=0.5,
        help="Fixed threshold for post-threshold mask evaluation (default: 0.5)",
    )
    parser.add_argument(
        "--allow-shape-mismatch",
        action="store_true",
        help="Skip mismatched shape pairs instead of raising an error",
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
        help="Optional cap for exported reconstructed volumes.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    output_dir = args.output_dir or (args.pred_dir / "evaluation_v2")
    if not args.pred_dir.exists():
        print(f"Error: prediction directory does not exist: {args.pred_dir}")
        sys.exit(1)
    if not args.gt_dir.exists():
        print(f"Error: ground truth directory does not exist: {args.gt_dir}")
        sys.exit(1)

    protocol = enforce_post_threshold_mode(make_fixed_protocol(args.fixed_threshold))

    matched, missing, total_gt = count_matched_pairs(args.pred_dir, args.gt_dir)
    print("=" * 60)
    print("EVALUATION V2 (nnU-Net post-threshold)")
    print("=" * 60)
    print(f"Predictions: {args.pred_dir}")
    print(f"Ground truth: {args.gt_dir}")
    print(f"Total GT files: {total_gt}")
    print(f"Matched pairs: {matched}")
    print(f"Missing predictions: {missing}")
    print(f"Threshold: {protocol.thresholds[0]:.4f}")
    print("=" * 60)

    samples = iter_nnunet_slice_samples(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        strict_shape=not args.allow_shape_mismatch,
    )
    engine = DualLevelStreamingMetricsEngine(
        thresholds=protocol.thresholds,
        assembler_case_key="nnunet_default",
    )
    export_dir = output_dir / "reconstructed_volumes" if args.export_reconstructed_volumes else None
    exported_count = 0
    for sample in samples:
        finalized_volumes = engine.update(sample)
        if args.export_reconstructed_volumes and finalized_volumes:
            remaining = _remaining_export_budget(exported_count, args.max_export_volumes_per_case)
            if remaining != 0:
                to_export = finalized_volumes if remaining < 0 else finalized_volumes[:remaining]
                export_reconstructed_volumes(
                    grouped_volumes={"nnunet_default": to_export},
                    output_dir=export_dir,
                    max_volumes_per_case=None,
                )
                exported_count += len(to_export)
    trailing_volumes = engine.finalize_open_volumes()
    if args.export_reconstructed_volumes and trailing_volumes:
        remaining = _remaining_export_budget(exported_count, args.max_export_volumes_per_case)
        if remaining != 0:
            to_export = trailing_volumes if remaining < 0 else trailing_volumes[:remaining]
            export_reconstructed_volumes(
                grouped_volumes={"nnunet_default": to_export},
                output_dir=export_dir,
                max_volumes_per_case=None,
            )
            exported_count += len(to_export)
    dual_level_results = engine.finalize()
    finalized_results = dual_level_results["slice_level"]
    volume_results = dual_level_results["volume_level"]
    selected_threshold = select_primary_threshold(finalized_results, protocol)

    payload = build_report_payload(
        finalized_results=finalized_results,
        protocol=protocol,
        entrypoint_name="compute_segmentation_metrics_for_nnunet_2d_predictions",
        metadata={
            "pred_dir": str(args.pred_dir.resolve()),
            "gt_dir": str(args.gt_dir.resolve()),
            "matched_pairs": matched,
            "missing_predictions": missing,
        },
        selected_threshold=selected_threshold,
        volume_finalized_results=volume_results,
    )

    json_path = write_json_report(payload, output_dir=output_dir)
    csv_path = write_threshold_csv(finalized_results, output_dir=output_dir)
    volume_csv_path = write_volume_threshold_csv(volume_results, output_dir=output_dir)
    summary_path = output_dir / "evaluation_summary.txt"
    summary_text = build_text_summary(payload)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary_text, encoding="utf-8")

    print("\nOutputs:")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print(f"  {volume_csv_path}")
    print(f"  {summary_path}")
    if args.export_reconstructed_volumes:
        print(f"  exported_reconstructed_volumes={exported_count} -> {export_dir / 'nnunet_default'}")
    print("-" * 60)
    print(summary_text)


def _remaining_export_budget(exported: int, max_per_case: int | None) -> int:
    if max_per_case is None:
        return -1
    remaining = int(max_per_case) - int(exported)
    return max(remaining, 0)


if __name__ == "__main__":
    main()

