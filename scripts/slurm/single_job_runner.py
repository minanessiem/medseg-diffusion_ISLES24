#!/usr/bin/env python3

import argparse
from typing import List
from datetime import datetime
import os

import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils.run_name import generate_run_name

from scripts.slurm.base_run_config import BASE_CONFIG, SLURM_TEMPLATE, update_logdir_paths
from scripts.slurm.job_runner import SlurmJobRunner
from scripts.slurm.utils.commandline_utils import add_config_arguments, update_config_from_args

def main():
    parser = argparse.ArgumentParser(description='Submit a single SLURM job for medseg-diffusion training')
    
    # Hydra-related arguments
    parser.add_argument('--config-name', type=str, default='cluster',
                        help='Hydra config name')
    parser.add_argument('--overrides', nargs='*', default=[],
                        help='Hydra overrides as key=value pairs (e.g., dataset.fold=0 training.max_epochs=300)')
    
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
    
    # Compose Hydra config
    with hydra.initialize(config_path="../../configs", version_base=None):
        cfg = hydra.compose(config_name=args.config_name, overrides=overrides)
    
    # Generate run_name using utility
    run_name = generate_run_name(cfg, timestamp)
    
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
