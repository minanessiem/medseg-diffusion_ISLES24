#!/usr/bin/env python3

import argparse
from typing import List
from datetime import datetime
import os

from scripts.slurm.base_run_config import BASE_CONFIG, SLURM_TEMPLATE, update_logdir_paths
from scripts.slurm.job_runner import SlurmJobRunner
from scripts.slurm.utils.commandline_utils import add_config_arguments, update_config_from_args

def generate_run_name(params: dict, timestamp: str) -> str:
    """Generate run_name mimicking main.py logic."""
    scheduler_str = ""
    if params.get('scheduler_type') == 'reduce_lr':
        sched_params = {
            'rlrfctr': params.get('reduce_lr_factor'),
            'rlrpat': params.get('reduce_lr_patience'),
            'rlrthrsh': params.get('reduce_lr_threshold'),
            'rlrcool': params.get('reduce_lr_cooldown')
        }
        for abbr, val in sched_params.items():
            if val is not None:
                scheduler_str += f"_{abbr}{val}"
    
    return (f"unet_img{params['image_size']}_numlayers{params['num_layers']}_firstconv{params['first_conv_channels']}_"
            f"timembdim{params['time_embedding_dim']}_attheads{params['att_heads']}_attheaddim{params['att_head_dim']}_"
            f"btllayers{params['bottleneck_transformer_layers']}_btchsz{params['train_batch_size']}_"
            f"lr{params['learning_rate']}_maxsteps{params['max_steps']}_diffsteps{params['timesteps']}"
            f"{scheduler_str}_{timestamp}")

def main():
    parser = argparse.ArgumentParser(description='Submit a single SLURM job for medseg-diffusion training')
    
    # Hydra-related arguments
    parser.add_argument('--config-name', type=str, default='cluster',
                        help='Hydra config name')
    parser.add_argument('--overrides', nargs='*', default=[],
                        help='Hydra overrides as key=value pairs (e.g., dataset.fold=0 training.max_epochs=300)')
    
    # Key parameters for run_name generation (matching main.py)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--first_conv_channels', type=int, default=8)
    parser.add_argument('--time_embedding_dim', type=int, default=128)
    parser.add_argument('--att_heads', type=int, default=2)
    parser.add_argument('--att_head_dim', type=int, default=2)
    parser.add_argument('--bottleneck_transformer_layers', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--max_steps', type=int, default=100000)
    parser.add_argument('--timesteps', type=int, default=100)
    parser.add_argument('--scheduler_type', type=str, default=None)
    parser.add_argument('--reduce_lr_factor', type=float, default=None)
    parser.add_argument('--reduce_lr_patience', type=int, default=None)
    parser.add_argument('--reduce_lr_threshold', type=float, default=None)
    parser.add_argument('--reduce_lr_cooldown', type=int, default=None)
    
    # Other args
    parser.add_argument('--dry-run', action='store_true',
                        help='Print job that would be submitted without submitting')
    parser.add_argument('--job-name', type=str, default=None,
                        help='Custom job name prefix (otherwise auto-generated)')
    
    # Add configuration override arguments from BASE_CONFIG
    add_config_arguments(parser, BASE_CONFIG)
    
    args = parser.parse_args()
    
    # Update BASE_CONFIG with command line arguments
    config = update_config_from_args(BASE_CONFIG.copy(), args, update_logdir_paths)
    
    # Parse overrides
    overrides = args.overrides  # Already a list from nargs='*'
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Collect params for run_name
    run_params = {
        'image_size': args.image_size,
        'num_layers': args.num_layers,
        'first_conv_channels': args.first_conv_channels,
        'time_embedding_dim': args.time_embedding_dim,
        'att_heads': args.att_heads,
        'att_head_dim': args.att_head_dim,
        'bottleneck_transformer_layers': args.bottleneck_transformer_layers,
        'train_batch_size': args.train_batch_size,
        'learning_rate': args.learning_rate,
        'max_steps': args.max_steps,
        'timesteps': args.timesteps,
        'scheduler_type': args.scheduler_type,
        'reduce_lr_factor': args.reduce_lr_factor,
        'reduce_lr_patience': args.reduce_lr_patience,
        'reduce_lr_threshold': args.reduce_lr_threshold,
        'reduce_lr_cooldown': args.reduce_lr_cooldown
    }
    
    # Generate run_name using main.py logic
    run_name = generate_run_name(run_params, timestamp)
    
    # Set job_name
    job_name = args.job_name or run_name
    
    # Update config
    config["job_name"] = job_name
    config["logdir_name"] = run_name  # For outputs
    config["hydra_config_name"] = args.config_name
    config["hydra_overrides"] = overrides
    
    # Update logdir paths with new logdir_name
    config = update_logdir_paths(config)
    
    # Initialize runner
    runner = SlurmJobRunner(config)
    
    # Submit
    runner.submit_job(config, SLURM_TEMPLATE, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
