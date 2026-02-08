#!/usr/bin/env python3
"""
Submit ISLES24 2D dataset conversion to nnU-Net format as a SLURM job.

This script wraps convert_isles24_2d_dataset_to_nnunet.py and submits it
as a non-interactive SLURM job.

Usage:
    python scripts/nnunet/slurm_runners/run_convert_isles24_2d_to_nnunet.py \
        [--time 02:00:00] [--cpus 64] [--mem 64G] \
        [--config-name convert_nnunet_cluster] \
        [hydra_overrides...] \
        [--dry-run]

Examples:
    # Default cluster conversion (test mode)
    python scripts/nnunet/slurm_runners/run_convert_isles24_2d_to_nnunet.py

    # Full export with custom fold
    python scripts/nnunet/slurm_runners/run_convert_isles24_2d_to_nnunet.py \
        nnunet.test=false dataset.fold=2

    # Custom resources then overrides
    python scripts/nnunet/slurm_runners/run_convert_isles24_2d_to_nnunet.py \
        --time 04:00:00 --cpus 32 --mem 64G \
        nnunet.test=false

    # Dry run to see what would be submitted
    python scripts/nnunet/slurm_runners/run_convert_isles24_2d_to_nnunet.py \
        nnunet.test=false --dry-run
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


# Default resource configuration for dataset conversion
# CPU/IO bound task - no heavy GPU usage needed
CONVERT_DEFAULTS = {
    "time": "02:00:00",
    "cpus_per_task": 64,
    "mem": "64G",
}


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Submit ISLES24 2D to nnU-Net conversion as SLURM job',
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
        default=CONVERT_DEFAULTS["time"],
        help=f'Time limit (default: {CONVERT_DEFAULTS["time"]})'
    )
    slurm_group.add_argument(
        '--cpus',
        type=int,
        default=CONVERT_DEFAULTS["cpus_per_task"],
        help=f'CPUs per task (default: {CONVERT_DEFAULTS["cpus_per_task"]})'
    )
    slurm_group.add_argument(
        '--mem',
        type=str,
        default=CONVERT_DEFAULTS["mem"],
        help=f'Memory allocation (default: {CONVERT_DEFAULTS["mem"]})'
    )
    
    # Configuration (second)
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument(
        '--config-name',
        type=str,
        default='convert_nnunet_cluster',
        help='Hydra config name (default: convert_nnunet_cluster)'
    )
    
    # Hydra overrides as positional (third)
    parser.add_argument(
        'hydra_overrides',
        nargs='*',
        help='Hydra config overrides (e.g., nnunet.test=false dataset.fold=2)'
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
    """Build the Python command to run dataset conversion."""
    cmd_parts = [
        'python3 -m scripts.nnunet.convert_isles24_2d_dataset_to_nnunet',
        f'--config-name={args.config_name}',
    ]
    
    # Add Hydra overrides
    if args.hydra_overrides:
        cmd_parts.extend(args.hydra_overrides)
    
    return ' '.join(cmd_parts)


def main():
    args = parse_arguments()
    
    # Build the Python command
    python_command = build_python_command(args)
    
    # Create job configuration
    config = BASE_CONFIG.copy()
    
    # Override with conversion-specific settings
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
    config['job_name'] = f'nnunet_convert_{timestamp}'
    
    # Set up log directory
    config['logdir_name'] = os.path.join(
        'nnunet_jobs', 'convert', 
        f'convert_{timestamp}'
    )
    config = update_logdir_paths(config)
    
    # Print job summary
    print("\n" + "=" * 60)
    print("ISLES24 2D → nnU-Net Conversion SLURM Job")
    print("=" * 60)
    print(f"  Config:      {args.config_name}")
    if args.hydra_overrides:
        print(f"  Overrides:   {' '.join(args.hydra_overrides)}")
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
