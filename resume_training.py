#!/usr/bin/env python3
"""
Resume training from a checkpoint.

Uses Hydra's Compose API to load the exact config that created the checkpoint,
then applies user overrides on top. This ensures reproducibility while allowing
modifications like extended training or adjusted learning rates.

Usage:
    python resume_training.py <run_dir> [--step STEP] [--config-name NAME] [overrides...]
    
Arguments:
    run_dir         Path to run directory containing checkpoints
    --step          Checkpoint step to resume from (int or "latest", default: latest)
    --config-name   Config name for legacy runs without .hydra/ (e.g., cluster_ddim250)
    overrides       Config overrides (e.g., training.max_steps=200000)

Examples:
    # Resume with same settings (latest checkpoint, requires .hydra/)
    python resume_training.py outputs/my_run_2024-01-01/
    
    # Resume with extended training
    python resume_training.py outputs/my_run_2024-01-01/ training.max_steps=200000
    
    # Resume legacy run without .hydra/ (provide config explicitly)
    python resume_training.py outputs/legacy_run/ --config-name cluster_ddim250_multitask
    
    # Resume from specific checkpoint step
    python resume_training.py outputs/my_run_2024-01-01/ --step 50000
    
    # Combine step selection with overrides
    python resume_training.py outputs/my_run_2024-01-01/ --step 50000 training.max_steps=200000

Notes:
    - For runs with .hydra/: config is loaded from .hydra/config.yaml
    - For legacy runs without .hydra/: use --config-name to specify config
    - Original overrides from .hydra/overrides.yaml are applied first (if present)
    - CLI overrides take precedence over original overrides
"""

# ============================================================================
# CRITICAL: Import MONAI FIRST, before any other imports
# MONAI must be imported before CUDA context is created or any library that
# might initialize CUDA (like TensorBoard/TensorFlow).
# ============================================================================
try:
    from monai.transforms import Resize, ScaleIntensityRange
    from monai.metrics import compute_hausdorff_distance, compute_surface_dice
    from monai.networks.utils import one_hot
except ImportError:
    pass  # MONAI not installed, that's OK for some configs

import argparse
import os
import sys

import hydra
from hydra import compose, initialize_config_dir, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(
        description='Resume training from checkpoint',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'run_dir', 
        help='Path to run directory containing checkpoints'
    )
    parser.add_argument(
        '--step', 
        type=str, 
        default='latest',
        help='Checkpoint step to resume from (int or "latest", default: latest)'
    )
    parser.add_argument(
        '--config-name',
        type=str,
        default=None,
        help='Config name for legacy runs without .hydra/ (e.g., cluster_ddim250_multitask)'
    )
    parser.add_argument(
        'overrides', 
        nargs='*', 
        help='Config overrides (e.g., training.max_steps=200000)'
    )
    args = parser.parse_args()
    
    # Validate run directory
    run_dir = os.path.abspath(args.run_dir)
    
    # Ensure trailing slash for consistency
    if not run_dir.endswith('/'):
        run_dir = run_dir + '/'
    
    hydra_dir = os.path.join(run_dir, '.hydra')
    has_hydra_dir = os.path.exists(hydra_dir)
    
    if not os.path.exists(run_dir):
        print(f"[ERROR] Run directory not found: {run_dir}")
        sys.exit(1)
    
    # Determine mode: standard (with .hydra/) or legacy (with --config-name)
    legacy_mode = False
    if not has_hydra_dir:
        if args.config_name:
            legacy_mode = True
            print(f"\n[LEGACY MODE] No .hydra/ found, using --config-name {args.config_name}")
        else:
            print(f"[ERROR] {hydra_dir} not found.")
            print("   Cannot resume without .hydra/ config directory.")
            print("   For legacy runs, use --config-name to specify the config:")
            print(f"   python resume_training.py {args.run_dir} --config-name <config>")
            sys.exit(1)
    
    # Sanitize overrides: convert '+key=' to '++key=' since we're re-applying
    def sanitize_override(override: str) -> str:
        if override.startswith('+') and not override.startswith('++'):
            return '+' + override  # Convert +key= to ++key=
        return override
    
    cli_overrides = [sanitize_override(o) for o in args.overrides]
    
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    
    if legacy_mode:
        # ====================================================================
        # LEGACY MODE: Load config from --config-name (like start_training.py)
        # ====================================================================
        print(f"\n[Resuming LEGACY run from: {run_dir}]")
        print(f"   Config: {args.config_name}")
        print(f"   Checkpoint step: {args.step}")
        if cli_overrides:
            print(f"   CLI overrides: {cli_overrides}")
        print(f"   Note: EMA will be initialized fresh, best metrics reset")
        
        # Use Hydra's normal initialization with config path
        config_path = os.path.abspath("configs")
        with initialize(version_base=None, config_path=os.path.relpath(config_path)):
            cfg = compose(config_name=args.config_name, overrides=cli_overrides)
        
    else:
        # ====================================================================
        # STANDARD MODE: Load config from .hydra/config.yaml
        # ====================================================================
        
        def is_config_group_reference(override: str) -> tuple:
            """
            Check if override is a config group reference like 'augmentation=aggressive_2d'.
            Returns (is_config_group, key, value) or (False, None, None).
            """
            if '=' not in override:
                return False, None, None
            
            # Strip any Hydra prefixes (+, ++, ~)
            clean = override.lstrip('+~')
            key, value = clean.split('=', 1)
            
            # Skip special keys that aren't config groups
            special_keys = {'timestamp', 'run_name', 'debug'}
            if key in special_keys:
                return False, None, None
            
            # If key contains a dot, it's a dotted path override, not a config group
            if '.' in key:
                return False, None, None
            
            # Check if there's a config directory for this key
            config_group_dir = os.path.join('configs', key)
            if os.path.isdir(config_group_dir):
                return True, key, value
            return False, None, None
        
        def load_config_group(group: str, name: str) -> dict:
            """Load a config group file and return as dict."""
            import yaml
            config_path = os.path.join('configs', group, f'{name}.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f) or {}
            else:
                raise FileNotFoundError(f"Config group file not found: {config_path}")
        
        # Separate CLI overrides into config groups and regular overrides
        cli_config_groups = []  # [(key, value), ...]
        cli_regular_overrides = []
        
        for override in cli_overrides:
            is_group, key, value = is_config_group_reference(override)
            if is_group:
                cli_config_groups.append((key, value))
            else:
                cli_regular_overrides.append(override)
        
        print(f"\n[Resuming from: {run_dir}]")
        print(f"   Config: {hydra_dir}/config.yaml")
        print(f"   Checkpoint step: {args.step}")
        if cli_config_groups:
            print(f"   CLI config groups: {cli_config_groups}")
        if cli_regular_overrides:
            print(f"   CLI overrides: {cli_regular_overrides}")
        
        # Load base config from .hydra/config.yaml (already has all original overrides resolved)
        # Only apply CLI regular overrides via Hydra
        with initialize_config_dir(version_base=None, config_dir=hydra_dir, job_name="resume"):
            cfg = compose(config_name="config", overrides=cli_regular_overrides)
        
        # Apply CLI config group overrides by loading and merging the config files
        for key, value in cli_config_groups:
            try:
                group_config = load_config_group(key, value)
                print(f"   Applying config group: {key}={value}")
                # Merge the loaded config into cfg[key]
                cfg[key] = OmegaConf.merge(cfg.get(key, {}), OmegaConf.create(group_config))
            except FileNotFoundError as e:
                print(f"   [WARNING] {e}")
                print(f"   Setting {key}={value} as string value")
                cfg[key] = value
    
    # Import and run training
    # Import here to avoid circular imports and ensure Hydra is initialized first
    from src.utils.train_utils import setup_config_aliases, setup_seeds, setup_and_resume_training
    
    # Set up config aliases (same as start_training.py)
    cfg = setup_config_aliases(cfg)
    
    # Set seeds for reproducibility
    setup_seeds(cfg)
    
    # Run resumed training (pass legacy_mode flag)
    setup_and_resume_training(cfg, run_dir=run_dir, resume_step=args.step, legacy_mode=legacy_mode)
    
    print(f"\n✓ Training resumed and completed successfully")


if __name__ == '__main__':
    main()

