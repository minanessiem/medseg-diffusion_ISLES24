#!/usr/bin/env python3
"""
Generic nnU-Net prediction evaluator (config-driven, 2D/3D aware).

This entrypoint composes:
- one conversion preset (nnunet/convert/*) for dataset/runtime context
- one evaluation policy preset (nnunet/eval/*) for evaluation behavior

and evaluates predictions in either:
- slices_2d mode (existing nnU-Net 2D naming), or
- volumes_3d mode (native per-case 3D volumes).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

from omegaconf import DictConfig, OmegaConf

from scripts.nnunet.core.evaluation_pipeline import run_nnunet_evaluation
from scripts.slurm.single_job_runner import apply_override, deep_merge, load_config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate nnU-Net predictions using composed convert/eval configs.",
    )
    parser.add_argument(
        "--convert-config-name",
        type=str,
        required=True,
        help="Conversion preset config path under configs/ (required).",
    )
    parser.add_argument(
        "--eval-config-name",
        type=str,
        required=True,
        help="Evaluation policy config path under configs/ (required).",
    )
    parser.add_argument(
        "--pred-dir",
        type=Path,
        default=None,
        help="Prediction directory override (highest precedence over config).",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=None,
        help="Ground-truth directory override (highest precedence over config).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Evaluation output directory override (highest precedence over config).",
    )
    parser.add_argument(
        "--input-format",
        type=str,
        choices=["slices_2d", "volumes_3d"],
        default=None,
        help="Input format override (default: config value).",
    )
    parser.add_argument(
        "--fixed-threshold",
        type=float,
        default=None,
        help="Fixed threshold override for post-threshold masks (default: config value).",
    )
    parser.add_argument(
        "--allow-shape-mismatch",
        action="store_true",
        help="Skip mismatched pairs instead of raising (overrides config).",
    )
    parser.add_argument(
        "--foreground-only-all-metrics",
        action="store_true",
        help="2D compatibility mode: apply foreground-only denominator to all reported metrics.",
    )
    parser.add_argument(
        "config_overrides",
        nargs="*",
        help="Hydra-style key=value overrides applied after config composition.",
    )
    return parser


def compose_runtime_config(
    convert_config_name: str,
    eval_config_name: str,
    config_overrides: List[str],
) -> DictConfig:
    """
    Compose convert + eval config trees and apply explicit key=value overrides.
    """
    convert_cfg = load_config(convert_config_name, [], resolve_final=True)
    eval_cfg = load_config(eval_config_name, [], resolve_final=True)

    composed: Dict[str, Any] = {}
    composed = deep_merge(composed, dict(convert_cfg))
    composed = deep_merge(composed, dict(eval_cfg))

    for override in config_overrides:
        composed = apply_override(composed, override)

    return OmegaConf.create(composed)


def apply_cli_overrides(cfg: DictConfig, args: argparse.Namespace) -> DictConfig:
    """
    Apply CLI flags as final-highest-precedence overrides.
    """
    cfg_dict: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise ValueError("Composed config must resolve to a dictionary.")

    nnunet_eval = cfg_dict.setdefault("nnunet_eval", {})
    if not isinstance(nnunet_eval, dict):
        raise ValueError("Composed nnunet_eval config must be a mapping.")

    if args.pred_dir is not None:
        nnunet_eval["pred_dir"] = str(args.pred_dir)
    if args.gt_dir is not None:
        nnunet_eval["gt_dir"] = str(args.gt_dir)
    if args.output_dir is not None:
        nnunet_eval["output_dir"] = str(args.output_dir)
    if args.input_format is not None:
        nnunet_eval["input_format"] = str(args.input_format)
    if args.fixed_threshold is not None:
        nnunet_eval["threshold"] = float(args.fixed_threshold)
    if args.allow_shape_mismatch:
        nnunet_eval["allow_shape_mismatch"] = True
    if args.foreground_only_all_metrics:
        nnunet_eval["foreground_only_all_metrics"] = True

    return OmegaConf.create(cfg_dict)


def main() -> None:
    args = build_arg_parser().parse_args()

    try:
        print(
            "[nnunet-eval] Composing configs: "
            f"convert={args.convert_config_name}, eval={args.eval_config_name}",
            flush=True,
        )
        cfg = compose_runtime_config(
            convert_config_name=args.convert_config_name,
            eval_config_name=args.eval_config_name,
            config_overrides=list(args.config_overrides),
        )
        print("[nnunet-eval] Applying CLI overrides (if provided)...", flush=True)
        cfg = apply_cli_overrides(cfg, args)

        print("[nnunet-eval] Running evaluation pipeline...", flush=True)
        results = run_nnunet_evaluation(
            cfg=cfg,
            convert_config_name=args.convert_config_name,
            eval_config_name=args.eval_config_name,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print("=" * 70)
    print("nnU-Net Evaluation")
    print("=" * 70)
    print(f"Input format:        {results['input_format']}")
    print(f"Levels:              {results['levels']}")
    print(f"Predictions:         {results['pred_dir']}")
    print(f"Ground truth:        {results['gt_dir']}")
    print(f"Matched pairs:       {results['matched_pairs']}/{results['total_gt_files']}")
    print(f"Missing predictions: {results['missing_predictions']}")
    print("-" * 70)
    print("Outputs:")
    print(f"  {results['json_path']}")
    if results["slice_csv_path"] is not None:
        print(f"  {results['slice_csv_path']}")
    if results["volume_csv_path"] is not None:
        print(f"  {results['volume_csv_path']}")
    print(f"  {results['summary_path']}")
    print("-" * 70)
    print(results["summary_text"])


if __name__ == "__main__":
    main()
