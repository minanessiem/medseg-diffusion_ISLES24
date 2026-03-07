#!/usr/bin/env python3
"""
Compute canonical segmentation metrics for nnU-Net 2D post-threshold predictions.
"""

import argparse
import sys
from pathlib import Path

from scripts.evaluation.io_nnunet import count_matched_pairs, iter_nnunet_slice_samples
from scripts.evaluation.metrics_engine import StreamingMetricsEngine
from scripts.evaluation.reporting import (
    build_report_payload,
    build_text_summary,
    write_json_report,
    write_threshold_csv,
)
from scripts.evaluation.threshold_protocol import (
    enforce_post_threshold_mode,
    make_fixed_protocol,
    select_primary_threshold,
)


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
    engine = StreamingMetricsEngine(thresholds=protocol.thresholds)
    finalized_results = engine.run(samples)
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
    )

    json_path = write_json_report(payload, output_dir=output_dir)
    csv_path = write_threshold_csv(finalized_results, output_dir=output_dir)
    summary_path = output_dir / "evaluation_summary.txt"
    summary_text = build_text_summary(payload)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary_text, encoding="utf-8")

    print("\nOutputs:")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print(f"  {summary_path}")
    print("-" * 60)
    print(summary_text)


if __name__ == "__main__":
    main()

