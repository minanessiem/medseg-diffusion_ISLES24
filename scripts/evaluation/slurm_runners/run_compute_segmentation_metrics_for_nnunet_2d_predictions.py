#!/usr/bin/env python3
"""
Submit evaluation-v2 nnU-Net metrics computation as a SLURM job.
"""

import argparse
import os
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
        description="Submit evaluation-v2 nnU-Net metrics computation as a SLURM job",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Script arguments
    parser.add_argument("--pred-dir", type=str, required=True, help="Directory with predicted *.nii.gz")
    parser.add_argument("--gt-dir", type=str, required=True, help="Directory with GT *.nii.gz")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory")
    parser.add_argument("--fixed-threshold", type=float, default=0.5, help="Fixed threshold value")
    parser.add_argument(
        "--allow-shape-mismatch",
        action="store_true",
        help="Skip mismatched prediction/GT shapes",
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
        help="Optional cap for exported reconstructed volumes.",
    )

    # SLURM resources
    parser.add_argument("--partition", type=str, default=None, help="SLURM partition override")
    parser.add_argument("--qos", type=str, default=None, help="SLURM QoS override")
    parser.add_argument("--time", type=str, default="00:30:00", help="Time limit")
    parser.add_argument("--cpus", type=int, default=32, help="CPUs per task")
    parser.add_argument("--mem", type=str, default="32G", help="Memory allocation")

    # Control
    parser.add_argument("--dry-run", action="store_true", help="Print configuration without submitting")

    return parser.parse_args()


def build_python_command(args: argparse.Namespace) -> str:
    cmd_parts = [
        "python3 -m scripts.evaluation.compute_segmentation_metrics_for_nnunet_2d_predictions",
        f"--pred-dir {args.pred_dir}",
        f"--gt-dir {args.gt_dir}",
        f"--fixed-threshold {args.fixed_threshold}",
    ]
    if args.output_dir:
        cmd_parts.append(f"--output-dir {args.output_dir}")
    if args.allow_shape_mismatch:
        cmd_parts.append("--allow-shape-mismatch")
    if args.export_reconstructed_volumes:
        cmd_parts.append("--export-reconstructed-volumes")
    if args.max_export_volumes_per_case is not None:
        cmd_parts.append(f"--max-export-volumes-per-case {args.max_export_volumes_per_case}")
    return " ".join(cmd_parts)


def main() -> None:
    args = parse_arguments()
    python_command = build_python_command(args)

    config = BASE_CONFIG.copy()
    config["python_command"] = python_command
    config["cpus_per_task"] = args.cpus
    config["mem"] = args.mem
    config["time"] = args.time
    if args.partition:
        config["partition"] = args.partition
    if args.qos:
        config["qos"] = args.qos

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["job_name"] = f"evalv2_nnunet_{timestamp}"
    config["logdir_name"] = os.path.join("evaluation_v2_jobs", "nnunet", f"run_{timestamp}")
    config = update_logdir_paths(config)

    print("\n" + "=" * 60)
    print("EVALUATION V2 NNUNET SLURM JOB")
    print("=" * 60)
    print(f"  Predictions: {args.pred_dir}")
    print(f"  Ground truth: {args.gt_dir}")
    print(f"  Threshold: {args.fixed_threshold}")
    print(f"  Allow shape mismatch: {args.allow_shape_mismatch}")
    print("-" * 60)
    print(f"  Command: {python_command}")
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

