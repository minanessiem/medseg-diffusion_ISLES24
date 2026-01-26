#!/usr/bin/env python3
"""
Submit threshold analysis job to SLURM cluster.

This script wraps the threshold analysis tool and submits it as a SLURM job.
It follows the same patterns as single_job_runner.py for consistency.

Usage:
    python scripts/slurm/run_threshold_analysis.py \
        --run-dir <path> \
        --model-name <name> \
        [--gpus 1] [--time 00:30:00] [--dry-run]

Examples:
    # Basic submission
    python scripts/slurm/run_threshold_analysis.py \
        --run-dir /mnt/outputs/discriminative_swinunetr/run_2026-01-17_10-30-00 \
        --model-name best_model_step_002000_dice_2d_fg_0.1815

    # With custom resources and dry-run
    python scripts/slurm/run_threshold_analysis.py \
        --run-dir /mnt/outputs/my_run \
        --model-name best_model \
        --gpus 1 \
        --time 00:45:00 \
        --dry-run

    # Custom threshold range
    python scripts/slurm/run_threshold_analysis.py \
        --run-dir /mnt/outputs/my_run \
        --model-name best_model \
        --thresholds 0.1:0.9:0.02
"""

import argparse
import os
import sys
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.slurm.base_run_config import BASE_CONFIG, SLURM_TEMPLATE, update_logdir_paths
from scripts.slurm.job_runner import SlurmJobRunner


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Submit threshold analysis job to SLURM cluster',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Analysis arguments (passed through to threshold_analysis.py)
    parser.add_argument(
        '--run-dir',
        required=True,
        help='Path to run directory containing checkpoints and .hydra config'
    )
    parser.add_argument(
        '--model-name',
        required=True,
        help='Model checkpoint name (without .pth extension)'
    )
    parser.add_argument(
        '--thresholds',
        type=str,
        default='0.05:0.95:0.05',
        help='Threshold range as "start:stop:step" or comma-separated (default: 0.05:0.95:0.05)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: auto-generated in run_dir/analysis/)'
    )
    parser.add_argument(
        '--save-per-sample',
        action='store_true',
        help='Save per-sample metrics CSV (can be large)'
    )
    parser.add_argument(
        '--num-visualizations',
        type=int,
        default=4,
        help='Number of comparison visualization images (default: 4)'
    )
    
    # SLURM resource arguments
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='Number of GPUs (default: 1)'
    )
    parser.add_argument(
        '--partition',
        type=str,
        default=None,
        help='SLURM partition (default: from BASE_CONFIG)'
    )
    parser.add_argument(
        '--qos',
        type=str,
        default=None,
        help='SLURM QoS (default: from BASE_CONFIG)'
    )
    parser.add_argument(
        '--cpus-per-task',
        type=int,
        default=16,
        help='CPUs per task (default: 16)'
    )
    parser.add_argument(
        '--mem',
        type=str,
        default='64G',
        help='Memory allocation (default: 64G)'
    )
    parser.add_argument(
        '--time',
        type=str,
        default='00:30:00',
        help='Time limit (default: 00:30:00)'
    )
    
    # Control arguments
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print job configuration without submitting'
    )
    
    return parser.parse_args()


def build_python_command(args: argparse.Namespace) -> str:
    """Build the Python command to run threshold analysis."""
    cmd_parts = [
        'python3 scripts/analysis/threshold_analysis.py',
        f'--run-dir {args.run_dir}',
        f'--model-name {args.model_name}',
        f'--thresholds {args.thresholds}',
        f'--num-visualizations {args.num_visualizations}',
    ]
    
    if args.output_dir:
        cmd_parts.append(f'--output-dir {args.output_dir}')
    
    if args.save_per_sample:
        cmd_parts.append('--save-per-sample')
    
    return ' '.join(cmd_parts)


def main():
    args = parse_arguments()
    
    # Build the Python command
    python_command = build_python_command(args)
    
    # Create job configuration
    config = BASE_CONFIG.copy()
    
    # Override with analysis-specific settings
    config['python_command'] = python_command
    config['gpus'] = args.gpus
    config['cpus_per_task'] = args.cpus_per_task
    config['mem'] = args.mem
    config['time'] = args.time
    
    if args.partition:
        config['partition'] = args.partition
    if args.qos:
        config['qos'] = args.qos
    
    # Create job name from model name (truncated for SLURM compatibility)
    model_name_short = args.model_name[:30] if len(args.model_name) > 30 else args.model_name
    config['job_name'] = f'thresh_{model_name_short}'
    
    # Extract relative path from container outputs to preserve full directory structure
    # e.g., /mnt/outputs/discriminative_swinunetr_170126/run_name/ -> discriminative_swinunetr_170126/run_name/
    container_outputs = config['container_outputs_dir'].rstrip('/')
    run_dir_clean = args.run_dir.rstrip('/')
    
    if run_dir_clean.startswith(container_outputs):
        # Get the relative path from outputs to the run dir (preserves parent folders)
        relative_run_path = run_dir_clean[len(container_outputs):].lstrip('/')
    else:
        # Fallback to basename if not under container outputs
        relative_run_path = os.path.basename(run_dir_clean)
    
    # Put SLURM logs inside the run directory's analysis folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config['logdir_name'] = os.path.join(relative_run_path, 'analysis', 'slurm_logs', f'threshold_{timestamp}')
    
    # Update logdir paths (converts to proper host paths like /dss/...)
    config = update_logdir_paths(config)
    
    # Print job summary
    print("\n" + "=" * 60)
    print("THRESHOLD ANALYSIS SLURM JOB")
    print("=" * 60)
    print(f"  Run dir:     {args.run_dir}")
    print(f"  Model:       {args.model_name}")
    print(f"  Thresholds:  {args.thresholds}")
    print(f"  Visualizations: {args.num_visualizations}")
    print("-" * 60)
    print("  Resources:")
    print(f"    GPUs:      {args.gpus}")
    print(f"    CPUs:      {args.cpus_per_task}")
    print(f"    Memory:    {args.mem}")
    print(f"    Time:      {args.time}")
    print(f"    Partition: {config['partition']}")
    print("-" * 60)
    print(f"  Python command:")
    print(f"    {python_command}")
    print("=" * 60)
    
    if args.dry_run:
        print("\n[DRY RUN] Would submit the following SLURM script:\n")
        
        # Show what the script would look like
        config['output_file'] = f"{config['host_logdir']}/output.out"
        config['error_file'] = f"{config['host_logdir']}/error.err"
        
        # Print key configuration
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

