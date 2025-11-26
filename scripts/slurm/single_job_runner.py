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
    """Apply a single key=value override, supporting dotted paths and config group references."""
    if '=' not in override:
        return cfg
    key_path, value = override.split('=', 1)
    keys = key_path.split('.')
    
    # Special handling for config group references (e.g., loss=mse_loss_only)
    # These should load the corresponding config file, not assign as string
    if len(keys) == 1:  # Simple key (no dots) - might be a config group
        config_group = keys[0]
        potential_file = f"configs/{config_group}/{value}.yaml"
        
        if os.path.exists(potential_file):
            # This is a config group reference - load the file
            with open(potential_file, 'r') as f:
                group_cfg = yaml.safe_load(f)
            
            # Recursively resolve defaults in the loaded config
            if 'defaults' in group_cfg:
                # Handle defaults (e.g., if mse_loss_only inherits from another config)
                for default in group_cfg['defaults']:
                    if isinstance(default, str) and default != '_self_':
                        # Load base config from same group
                        base_path = f"configs/{config_group}/{default}.yaml"
                        if os.path.exists(base_path):
                            with open(base_path, 'r') as f:
                                base_cfg = yaml.safe_load(f)
                            group_cfg = deep_merge(base_cfg, group_cfg)
                del group_cfg['defaults']  # Clean up
            
            # Resolve any interpolations in the loaded config
            group_cfg = resolve_interpolations(group_cfg, cfg)
            
            # Merge into main config
            cfg[config_group] = group_cfg
            return cfg
    
    # Standard dotted path override (existing logic)
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

def resolve_interpolations(cfg: Dict, root: Dict = None) -> Dict:
    """Recursively resolve ${path.to.key} interpolations in the dict."""
    if root is None:
        root = cfg
    if isinstance(cfg, dict):
        for key, value in cfg.items():
            cfg[key] = resolve_interpolations(value, root)
        return cfg
    elif isinstance(cfg, str) and cfg.startswith('${') and cfg.endswith('}'):
        path = cfg[2:-1].split('.')
        current = root
        for p in path:
            if isinstance(current, dict) and p in current:
                current = current[p]
            else:
                raise KeyError(f"Interpolation key '{cfg}' not found")
        return current
    return cfg

def load_config(config_name: str, overrides: List[str]) -> Dict:
    """Load and merge YAML configs mimicking basic Hydra composition."""
    config_path = f"configs/{config_name}.yaml"  # Relative from project root
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Merge defaults with nesting (recursively)
    if 'defaults' in cfg:
        for default in cfg['defaults']:
            if isinstance(default, str):
                # Skip Hydra's special _self_ keyword (controls merge order)
                if default == '_self_':
                    continue
                # Simple string reference: load another config file recursively
                # e.g., '- local' or '- cluster'
                sub_cfg = load_config(default, [])  # Recursive call
                cfg = deep_merge(cfg, sub_cfg)
            elif isinstance(default, dict):
                key, file_name = next(iter(default.items()))
                
                # Handle Hydra's 'override /section: file' syntax
                is_override = False
                if key.startswith('override /'):
                    is_override = True
                    section = key[len('override /'):].strip()  # Strip 'override /' prefix
                else:
                    section = key
                
                sub_path = f"configs/{section}/{file_name}.yaml"
                with open(sub_path, 'r') as f:
                    sub_cfg = yaml.safe_load(f)
                
                # Override replaces, normal default merges
                if is_override:
                    # Override: replace entire section
                    cfg[section] = sub_cfg
                else:
                    # Normal default: merge if section exists, else assign
                    if section in cfg and isinstance(cfg[section], dict):
                        cfg[section] = deep_merge(cfg[section], sub_cfg)
                    else:
                        cfg[section] = sub_cfg
        del cfg['defaults']  # Clean up
    
    # Apply overrides
    for ovr in overrides:
        cfg = apply_override(cfg, ovr)
    
    # Resolve interpolations
    cfg = resolve_interpolations(cfg)
    
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
    
    # SLURM resource arguments
    parser.add_argument('--gpus', type=int, default=None,
                        help='Number of GPUs to request (overrides BASE_CONFIG)')
    parser.add_argument('--partition', type=str, default=None,
                        help='SLURM partition to use (overrides BASE_CONFIG)')
    parser.add_argument('--cpus-per-task', type=int, default=None,
                        help='CPUs per task (overrides BASE_CONFIG)')
    parser.add_argument('--mem', type=str, default=None,
                        help='Memory allocation (e.g., "256G", overrides BASE_CONFIG)')
    
    # Add configuration override arguments from BASE_CONFIG
    add_config_arguments(parser, BASE_CONFIG)
    
    args = parser.parse_args()
    
    # Update BASE_CONFIG with command line arguments
    config = update_config_from_args(BASE_CONFIG.copy(), args, update_logdir_paths)
    
    # Override SLURM resource parameters if specified
    if args.gpus is not None:
        config["gpus"] = args.gpus
    if args.partition is not None:
        config["partition"] = args.partition
    if args.cpus_per_task is not None:
        config["cpus_per_task"] = args.cpus_per_task
    if args.mem is not None:
        config["mem"] = args.mem
    
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
    
    # Add timestamp and run_name to overrides for main.py
    overrides.append(f"timestamp={timestamp}")
    overrides.append(f"run_name={run_name}")
    config["hydra_overrides"] = overrides
    
    # Update config with resolved output_root
    config["container_outputs_dir"] = cfg["environment"]["training"]["output_root"]
    # Derive relative part (strip container's /mnt/)
    if config["container_outputs_dir"].startswith(config["container_prefix"]):
        relative_out = config["container_outputs_dir"][len(config["container_prefix"]):]
        config["host_outputs_dir"] = config["host_base"] + relative_out
    else:
        raise ValueError(f"output_root '{{config[\"container_outputs_dir\"]}}' does not start with expected container_prefix '{{config[\"container_prefix\"]}}'")

    # Update logdir paths with new logdir_name
    config = update_logdir_paths(config)
    
    # Initialize runner
    runner = SlurmJobRunner(config)
    
    # Submit
    runner.submit_job(config, SLURM_TEMPLATE, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
