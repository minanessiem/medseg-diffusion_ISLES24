#!/usr/bin/env python3
"""
Submit nnU-Net 2D prediction metrics computation as a SLURM job.

This script wraps compute_segmentation_metrics_for_nnunet_2d_predictions.py
and submits it as a non-interactive SLURM job.

Usage:
    python scripts/nnunet/slurm_runners/run_compute_segmentation_metrics_for_nnunet_2d_predictions.py \
        [--time 00:30:00] [--cpus 32] [--mem 32G] \
        --pred-dir <path> --gt-dir <path> \
        [--dry-run]

Examples:
    # Evaluate predictions against ground truth labels
    python scripts/nnunet/slurm_runners/run_compute_segmentation_metrics_for_nnunet_2d_predictions.py \
        --pred-dir /mnt/outputs/nnunet_results/predictions \
        --gt-dir /mnt/datasets/nnunet_raw/Dataset050_isles24/labelsTs

    # Custom resources
    python scripts/nnunet/slurm_runners/run_compute_segmentation_metrics_for_nnunet_2d_predictions.py \
        --time 01:00:00 --mem 64G \
        --pred-dir /mnt/outputs/nnunet_results/predictions \
        --gt-dir /mnt/datasets/nnunet_raw/Dataset050_isles24/labelsTs

    # Dry run to see what would be submitted
    python scripts/nnunet/slurm_runners/run_compute_segmentation_metrics_for_nnunet_2d_predictions.py \
        --pred-dir /mnt/outputs/nnunet_results/predictions \
        --gt-dir /mnt/datasets/nnunet_raw/Dataset050_isles24/labelsTs \
        --dry-run
"""

import argparse
import os
import sys
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.slurm.base_run_config import BASE_CONFIG, SLURM_TEMPLATE, update_logdir_paths
from scripts.slurm.job_runner import SlurmJobRunner


# Default resource configuration for metrics computation
# Lightweight CPU/IO task - loads NIfTI files and computes torch metrics
METRICS_DEFAULTS = {
    "time": "00:30:00",
    "cpus_per_task": 32,
    "mem": "32G",
}


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Submit nnU-Net 2D prediction metrics computation as SLURM job',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # SLURM resource arguments (first)
    slurm_group = parser.add_argument_group('SLURM resources')
    slurm_group.add_argument(
        '--partition',
        type=str,
        default=None,
        help=f'SLURM partition (default: {BASE_CONFIG["partition"]})'
    )
    slurm_group.add_argument(
        '--qos',
        type=str,
        default=None,
        help=f'SLURM QoS (default: {BASE_CONFIG["qos"]})'
    )
    slurm_group.add_argument(
        '--time',
        type=str,
        default=METRICS_DEFAULTS["time"],
        help=f'Time limit (default: {METRICS_DEFAULTS["time"]})'
    )
    slurm_group.add_argument(
        '--cpus',
        type=int,
        default=METRICS_DEFAULTS["cpus_per_task"],
        help=f'CPUs per task (default: {METRICS_DEFAULTS["cpus_per_task"]})'
    )
    slurm_group.add_argument(
        '--mem',
        type=str,
        default=METRICS_DEFAULTS["mem"],
        help=f'Memory allocation (default: {METRICS_DEFAULTS["mem"]})'
    )
    
    # Script arguments (second — pass-through)
    script_group = parser.add_argument_group('Metrics script arguments')
    script_group.add_argument(
        '--pred-dir',
        type=str,
        required=True,
        help='Directory containing nnU-Net prediction NIfTI files (*.nii.gz)'
    )
    script_group.add_argument(
        '--gt-dir',
        type=str,
        required=True,
        help='Directory containing ground truth label NIfTI files (*.nii.gz)'
    )
    
    # Control arguments (last)
    control_group = parser.add_argument_group('Control')
    control_group.add_argument(
        '--dry-run',
        action='store_true',
        help='Print job configuration without submitting'
    )
    
    return parser.parse_args()


def build_python_command(args: argparse.Namespace) -> str:
    """Build the Python command to run metrics computation."""
    cmd_parts = [
        'python3 -m scripts.nnunet.compute_segmentation_metrics_for_nnunet_2d_predictions',
        f'--pred-dir {args.pred_dir}',
        f'--gt-dir {args.gt_dir}',
    ]
    
    return ' '.join(cmd_parts)


def main():
    args = parse_arguments()
    
    # Build the Python command
    python_command = build_python_command(args)
    
    # Create job configuration
    config = BASE_CONFIG.copy()
    
    # Override with metrics-specific settings
    config['python_command'] = python_command
    config['cpus_per_task'] = args.cpus
    config['mem'] = args.mem
    config['time'] = args.time
    
    if args.partition:
        config['partition'] = args.partition
    if args.qos:
        config['qos'] = args.qos
    
    # Create job name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config['job_name'] = f'nnunet_metrics_{timestamp}'
    
    # Set up log directory
    config['logdir_name'] = os.path.join(
        'nnunet_jobs', 'metrics',
        f'metrics_{timestamp}'
    )
    config = update_logdir_paths(config)
    
    # Print job summary
    print("\n" + "=" * 60)
    print("Segmentation Metrics for nnU-Net 2D Predictions SLURM Job")
    print("=" * 60)
    print(f"  Predictions: {args.pred_dir}")
    print(f"  Ground Truth: {args.gt_dir}")
    print("-" * 60)
    print("  Resources:")
    print(f"    Partition: {config['partition']}")
    print(f"    GPUs:      {config['gpus']}")
    print(f"    CPUs:      {args.cpus}")
    print(f"    Memory:    {args.mem}")
    print(f"    Time:      {args.time}")
    print("-" * 60)
    print(f"  Command:")
    print(f"    {python_command}")
    print("=" * 60)
    
    if args.dry_run:
        print("\n[DRY RUN] Would submit the following SLURM job:\n")
        
        # Show what the script would look like
        config['output_file'] = f"{config['host_logdir']}/output.out"
        config['error_file'] = f"{config['host_logdir']}/error.err"
        
        print("Configuration:")
        for key in ['job_name', 'partition', 'qos', 'gpus', 'time',
                    'cpus_per_task', 'mem', 'host_logdir', 'python_command']:
            print(f"  {key}: {config[key]}")
        
        print("\n[DRY RUN] No job submitted.")
        return
    
    # Submit the job
    runner = SlurmJobRunner(config)
    job_id = runner.submit_job(config, SLURM_TEMPLATE, dry_run=False)
    
    if job_id:
        print(f"\n✓ Job submitted successfully: {job_id}")
        print(f"  Monitor with: squeue -j {job_id}")
        print(f"  Logs at: {config['host_logdir']}")
    else:
        print("\n✗ Job submission failed")
        sys.exit(1)


if __name__ == '__main__':
    main()

