#!/usr/bin/env python3

import argparse
from typing import List
from datetime import datetime
import os
import yaml
from typing import Dict
import pprint

from src.utils.run_name import generate_run_name

from scripts.slurm.base_run_config import BASE_CONFIG, SLURM_TEMPLATE, update_logdir_paths
from scripts.slurm.job_runner import SlurmJobRunner
from scripts.slurm.utils.commandline_utils import add_config_arguments, update_config_from_args

def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
    """Recursively merge dict2 into dict1."""
    for key, value in dict2.items():
        if isinstance(value, dict) and key in dict1 and isinstance(dict1[key], dict):
            dict1[key] = deep_merge(dict1[key], value)
        else:
            dict1[key] = value
    return dict1

def apply_override(cfg: Dict, override: str) -> Dict:
    """Apply a single key=value override, supporting dotted paths."""
    if '=' not in override:
        return cfg
    key_path, value = override.split('=', 1)
    keys = key_path.split('.')
    current = cfg
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    # Convert value to int/float if possible
    try:
        value = int(value)
    except ValueError:
        try:
            value = float(value)
        except ValueError:
            pass  # Keep as string
    current[keys[-1]] = value
    return cfg

def load_config(config_name: str, overrides: List[str]) -> Dict:
    """Load and merge YAML configs mimicking basic Hydra composition."""
    config_path = f"configs/{config_name}.yaml"  # Relative from project root
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Merge defaults with nesting
    if 'defaults' in cfg:
        for default in cfg['defaults']:
            if isinstance(default, dict):
                section, file_name = next(iter(default.items()))
                sub_path = f"configs/{section}/{file_name}.yaml"
                with open(sub_path, 'r') as f:
                    sub_cfg = yaml.safe_load(f)
                # Consistent merge for all - deep_merge if section exists, else assign
                if section in cfg and isinstance(cfg[section], dict):
                    cfg[section] = deep_merge(cfg[section], sub_cfg)
                else:
                    cfg[section] = sub_cfg
        del cfg['defaults']  # Clean up
    
    # Apply overrides
    for ovr in overrides:
        cfg = apply_override(cfg, ovr)
    
    return cfg

def main():
    parser = argparse.ArgumentParser(description='Submit a single SLURM job for medseg-diffusion training')
    
    # Hydra-related arguments (reused for custom loader)
    parser.add_argument('--config-name', type=str, default='cluster',
                        help='Config name (loads configs/{config-name}.yaml)')
    parser.add_argument('--overrides', nargs='*', default=[],
                        help='Overrides as key=value pairs (e.g., model.image_size=64)')
    
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
    overrides = args.overrides
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Load custom config
    cfg = load_config(args.config_name, overrides)
    pprint.pprint(cfg)  # Debug print of final config
    
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
