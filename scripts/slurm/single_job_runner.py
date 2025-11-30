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
    
    # Parse value: handle Hydra list syntax [val1,val2,...] and primitives
    if value.startswith('[') and value.endswith(']'):
        # Hydra list syntax: [0,1,2,3] or [true,false]
        list_content = value[1:-1].strip()
        if list_content:
            items = [item.strip() for item in list_content.split(',')]
            parsed_items = []
            for item in items:
                # Try to parse each item as int/float/bool
                try:
                    parsed_items.append(int(item))
                except ValueError:
                    try:
                        parsed_items.append(float(item))
                    except ValueError:
                        if item.lower() == 'true':
                            parsed_items.append(True)
                        elif item.lower() == 'false':
                            parsed_items.append(False)
                        else:
                            parsed_items.append(item)  # Keep as string
            value = parsed_items
        else:
            value = []  # Empty list
    else:
        # Try to convert single value to int/float/bool
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                # else: Keep as string
    
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
    
    # Resume support
    parser.add_argument('--resume-dir', type=str, default=None,
                        help='Container path to run directory to resume (e.g., /mnt/outputs/my_run/). '
                             'When set, uses resume_training.py instead of start_training.py.')
    
    # Other args
    parser.add_argument('--dry-run', action='store_true',
                        help='Print job that would be submitted without submitting')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging (adds debug=true to training overrides)')
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
    parser.add_argument('--time', type=str, default=None,
                        help='Time limit (e.g., "00:30:00" for 30 min, overrides BASE_CONFIG)')
    
    # Add configuration override arguments from BASE_CONFIG (exclude already-added params)
    excluded_params = {'gpus', 'partition', 'cpus_per_task', 'mem', 'time'}
    filtered_config = {k: v for k, v in BASE_CONFIG.items() if k not in excluded_params}
    add_config_arguments(parser, filtered_config)
    
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
    if args.time is not None:
        config["time"] = args.time
    
    # Parse overrides
    overrides = list(args.overrides)  # Make a copy to avoid modifying original
    
    # Debug mode: add debug=true to overrides for training script
    if args.debug:
        overrides.append("debug=true")
        print(f"[DEBUG] Debug mode enabled - added debug=true to overrides")
    
    def _debug(msg):
        if args.debug:
            print(f"[DEBUG:single_job_runner] {msg}")
    
    _debug("="*60)
    _debug("single_job_runner.py started")
    _debug("="*60)
    _debug(f"config-name: {args.config_name}")
    _debug(f"resume-dir: {args.resume_dir}")
    _debug(f"overrides: {overrides}")
    _debug(f"dry-run: {args.dry_run}")
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _debug(f"timestamp: {timestamp}")
    
    # ==========================================================================
    # Branch: Resume mode vs Fresh training mode
    # ==========================================================================
    if args.resume_dir:
        # ======================================================================
        # RESUME MODE: Use resume_training.py with existing run directory
        # ======================================================================
        _debug("Entering RESUME MODE branch")
        resume_dir = args.resume_dir.rstrip('/')
        _debug(f"resume_dir (stripped): {resume_dir}")
        
        # Extract run_name from the resume directory path
        run_name = os.path.basename(resume_dir)
        _debug(f"run_name extracted: {run_name}")
        
        # Set job_name (same as original run)
        job_name = args.job_name or run_name
        _debug(f"job_name: {job_name}")
        
        print(f"\n{'='*60}")
        print(f"RESUME MODE")
        print(f"{'='*60}")
        print(f"  Resume directory: {resume_dir}")
        print(f"  Run name: {run_name}")
        print(f"  Job name: {job_name}")
        if overrides:
            print(f"  Overrides: {overrides}")
        print(f"{'='*60}\n")
        
        # Update config for resume
        config["job_name"] = job_name
        config["logdir_name"] = run_name
        _debug(f"config[job_name]: {config['job_name']}")
        _debug(f"config[logdir_name]: {config['logdir_name']}")
        
        # Build python command for resume_training.py
        overrides_str = ' '.join(overrides) if overrides else ''
        config["python_command"] = f"python3 resume_training.py {resume_dir} {overrides_str}".strip()
        _debug(f"python_command: {config['python_command']}")
        
        # Set container logdir to the resume directory
        config["container_logdir"] = resume_dir
        _debug(f"container_logdir: {config['container_logdir']}")
        
        # Derive host logdir from container path
        # resume_dir format: /mnt/outputs/run_name or similar
        if resume_dir.startswith(config["container_outputs_base"].rstrip('/')):
            relative_path = resume_dir[len(config["container_outputs_base"].rstrip('/')):]
            config["host_logdir"] = os.path.join(config["host_outputs_dir"], relative_path.lstrip('/'))
            _debug(f"host_logdir (derived): {config['host_logdir']}")
        else:
            # Fallback: use the run_name under host_outputs_dir
            config["host_logdir"] = os.path.join(config["host_outputs_dir"], run_name)
            _debug(f"host_logdir (fallback): {config['host_logdir']}")
        
    else:
        # ======================================================================
        # FRESH TRAINING MODE: Use start_training.py with new run directory
        # ======================================================================
        _debug("Entering FRESH TRAINING MODE branch")
        
        # Load custom config
        _debug(f"Loading config: {args.config_name}")
        cfg = load_config(args.config_name, overrides)
        pprint.pprint(cfg)  # Debug print of final config
        _debug("Config loaded successfully")
        
        # Generate run_name using utility
        _debug("Generating run_name...")
        run_name = generate_run_name(cfg, timestamp)
        _debug(f"run_name: {run_name}")
        
        # Set job_name
        job_name = args.job_name or run_name
        _debug(f"job_name: {job_name}")
        
        print(f"\n{'='*60}")
        print(f"FRESH TRAINING MODE")
        print(f"{'='*60}")
        print(f"  Config: {args.config_name}")
        print(f"  Run name: {run_name}")
        print(f"  Job name: {job_name}")
        if overrides:
            print(f"  Overrides: {overrides}")
        print(f"{'='*60}\n")
        
        # Update config
        config["job_name"] = job_name
        config["logdir_name"] = run_name  # For outputs
        config["hydra_config_name"] = args.config_name
        _debug(f"config[job_name]: {config['job_name']}")
        _debug(f"config[logdir_name]: {config['logdir_name']}")
        
        # Add timestamp and run_name to overrides for start_training.py
        all_overrides = overrides + [f"timestamp={timestamp}", f"run_name={run_name}"]
        _debug(f"all_overrides: {all_overrides}")
        
        # Build python command for start_training.py
        overrides_str = ' '.join(all_overrides) if all_overrides else ''
        config["python_command"] = f"python3 start_training.py --config-name {args.config_name} {overrides_str}".strip()
        _debug(f"python_command: {config['python_command']}")
        
        # Update config with resolved output_root
        config["container_outputs_dir"] = cfg["environment"]["training"]["output_root"]
        _debug(f"container_outputs_dir: {config['container_outputs_dir']}")
        
        # Derive relative part (strip container's /mnt/)
        if config["container_outputs_dir"].startswith(config["container_prefix"]):
            relative_out = config["container_outputs_dir"][len(config["container_prefix"]):]
            config["host_outputs_dir"] = config["host_base"] + relative_out
            _debug(f"host_outputs_dir: {config['host_outputs_dir']}")
        else:
            raise ValueError(f"output_root '{{config[\"container_outputs_dir\"]}}' does not start with expected container_prefix '{{config[\"container_prefix\"]}}'")

        # Update logdir paths with new logdir_name
        config = update_logdir_paths(config)
        _debug(f"host_logdir: {config.get('host_logdir')}")
        _debug(f"container_logdir: {config.get('container_logdir')}")
    
    # ==========================================================================
    # Common: Submit the job
    # ==========================================================================
    _debug("="*60)
    _debug("Preparing to submit job")
    _debug("="*60)
    _debug(f"Final python_command: {config.get('python_command')}")
    _debug(f"Final host_logdir: {config.get('host_logdir')}")
    _debug(f"Final container_logdir: {config.get('container_logdir')}")
    
    # Initialize runner
    runner = SlurmJobRunner(config)
    _debug("SlurmJobRunner initialized")
    
    # Submit
    _debug(f"Calling submit_job (dry_run={args.dry_run})...")
    runner.submit_job(config, SLURM_TEMPLATE, dry_run=args.dry_run)
    _debug("submit_job complete")

if __name__ == "__main__":
    main()
