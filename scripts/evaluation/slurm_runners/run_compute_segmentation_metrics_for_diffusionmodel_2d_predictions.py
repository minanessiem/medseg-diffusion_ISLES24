#!/usr/bin/env python3
"""
Submit evaluation-v2 custom model metrics computation as a SLURM job.
"""

import argparse
import os
import shlex
import sys
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.slurm.base_run_config import BASE_CONFIG, SLURM_TEMPLATE, update_logdir_paths
from scripts.slurm.job_runner import SlurmJobRunner


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit evaluation-v2 custom model metrics computation as a SLURM job",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Script arguments
    parser.add_argument("--run-dir", required=True, help="Run directory with checkpoints and .hydra config")
    parser.add_argument("--model-name", required=True, help="Model checkpoint name without .pth extension")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory for reports")
    parser.add_argument("--ema", action="store_true", help="Use EMA checkpoint")
    parser.add_argument("--thresholds", type=str, default="0.05:0.95:0.05", help='Sweep thresholds "start:stop:step"')
    parser.add_argument("--fixed-threshold", type=float, default=None, help="If set, skip sweep and use one threshold")
    parser.add_argument("--optimize-metric", type=str, default="dice", help="Metric used to choose primary threshold")
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
        help="Prediction merge method",
    )
    parser.add_argument("--staple-max-iters", type=int, default=5, help="STAPLE max iterations")
    parser.add_argument("--staple-tolerance", type=float, default=0.02, help="STAPLE tolerance")
    parser.add_argument("--test", action="store_true", help="Quick test mode (small number of slices)")
    parser.add_argument(
        "--test-max-slices",
        type=int,
        default=10,
        help="Number of slices to process in --test mode",
    )
    parser.add_argument(
        "--export-reconstructed-volumes",
        action="store_true",
        help="Export reconstructed prediction/GT volumes as NIfTI.",
    )
    parser.add_argument(
        "--max-export-volumes-per-case",
        type=int,
        default=None,
        help="Optional cap for exported reconstructed volumes per analysis case.",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help=(
            "Optional config overrides forwarded to eval script. "
            "Examples: distribution=dp distribution.timeout_minutes=60"
        ),
    )

    # SLURM resources
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--partition", type=str, default=None, help="SLURM partition override")
    parser.add_argument("--qos", type=str, default=None, help="SLURM QoS override")
    parser.add_argument("--cpus-per-task", type=int, default=32, help="CPUs per task")
    parser.add_argument("--mem", type=str, default="64G", help="Memory allocation")
    parser.add_argument("--time", type=str, default="02:00:00", help="Time limit")

    # Control
    parser.add_argument("--dry-run", action="store_true", help="Print configuration without submitting")

    return parser.parse_args()


def build_python_command(args: argparse.Namespace) -> str:
    cmd_parts = [
        "python3 -m scripts.evaluation.compute_segmentation_metrics_for_diffusionmodel_2d_predictions",
        f"--run-dir {shlex.quote(args.run_dir)}",
        f"--model-name {shlex.quote(args.model_name)}",
        f"--optimize-metric {shlex.quote(args.optimize_metric)}",
        f"--ensemble-samples {shlex.quote(args.ensemble_samples)}",
        f"--ensemble-method {shlex.quote(args.ensemble_method)}",
        f"--staple-max-iters {args.staple_max_iters}",
        f"--staple-tolerance {args.staple_tolerance}",
    ]

    if args.ema:
        cmd_parts.append("--ema")
    if args.fixed_threshold is not None:
        cmd_parts.append(f"--fixed-threshold {args.fixed_threshold}")
    else:
        cmd_parts.append(f"--thresholds {shlex.quote(args.thresholds)}")
    if args.output_dir:
        cmd_parts.append(f"--output-dir {shlex.quote(args.output_dir)}")
    if args.test:
        cmd_parts.append("--test")
        cmd_parts.append(f"--test-max-slices {args.test_max_slices}")
    if args.export_reconstructed_volumes:
        cmd_parts.append("--export-reconstructed-volumes")
    if args.max_export_volumes_per_case is not None:
        cmd_parts.append(f"--max-export-volumes-per-case {args.max_export_volumes_per_case}")
    if args.overrides:
        cmd_parts.append("--overrides")
        cmd_parts.extend(shlex.quote(override) for override in args.overrides)

    return " ".join(cmd_parts)


def main() -> None:
    args = parse_arguments()
    python_command = build_python_command(args)

    config = BASE_CONFIG.copy()
    config["python_command"] = python_command
    config["gpus"] = args.gpus
    config["cpus_per_task"] = args.cpus_per_task
    config["mem"] = args.mem
    config["time"] = args.time
    if args.partition:
        config["partition"] = args.partition
    if args.qos:
        config["qos"] = args.qos

    model_name_short = args.model_name[:26] if len(args.model_name) > 26 else args.model_name
    config["job_name"] = f"evalv2_{model_name_short}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["logdir_name"] = os.path.join("evaluation_v2_jobs", "diffusion", f"run_{timestamp}")
    config = update_logdir_paths(config)

    print("\n" + "=" * 60)
    print("EVALUATION V2 CUSTOM MODEL SLURM JOB")
    print("=" * 60)
    print(f"  Run dir:    {args.run_dir}")
    print(f"  Model:      {args.model_name}")
    print(f"  EMA:        {args.ema}")
    if args.fixed_threshold is not None:
        print(f"  Protocol:   fixed (τ={args.fixed_threshold})")
    else:
        print(f"  Protocol:   sweep ({args.thresholds}), optimize={args.optimize_metric}")
    print(f"  Ensemble:   n={args.ensemble_samples}, method={args.ensemble_method}")
    if args.test:
        print(f"  Test mode:  enabled (max_slices={args.test_max_slices})")
    if args.overrides:
        print(f"  Overrides:  {args.overrides}")
    print("-" * 60)
    print(f"  Command:    {python_command}")
    print("=" * 60)

    if args.dry_run:
        config["output_file"] = f"{config['host_logdir']}/output.out"
        config["error_file"] = f"{config['host_logdir']}/error.err"
        print("\n[DRY RUN] Configuration:")
        for key in [
            "job_name",
            "partition",
            "qos",
            "gpus",
            "time",
            "cpus_per_task",
            "mem",
            "host_logdir",
            "python_command",
        ]:
            print(f"  {key}: {config[key]}")
        print("\n[DRY RUN] No job submitted.")
        return

    runner = SlurmJobRunner(config)
    job_id = runner.submit_job(config, SLURM_TEMPLATE, dry_run=False)
    if job_id:
        print(f"\n✓ Job submitted successfully: {job_id}")
        print(f"  Monitor with: squeue -j {job_id}")
        print(f"  Logs at: {config['host_logdir']}")
    else:
        print("\n✗ Job submission failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

