#!/usr/bin/env python3
"""
Resume training from a checkpoint.

Uses Hydra's Compose API to load the exact config that created the checkpoint,
then applies user overrides on top. This ensures reproducibility while allowing
modifications like extended training or adjusted learning rates.

Usage:
    python resume_training.py <run_dir> [--step STEP] [overrides...]
    
Arguments:
    run_dir     Path to run directory containing .hydra/ and checkpoints
    --step      Checkpoint step to resume from (int or "latest", default: latest)
    overrides   Config overrides (e.g., training.max_steps=200000)

Examples:
    # Resume with same settings (latest checkpoint)
    python resume_training.py outputs/my_run_2024-01-01/
    
    # Resume with extended training
    python resume_training.py outputs/my_run_2024-01-01/ training.max_steps=200000
    
    # Resume with different learning rate
    python resume_training.py outputs/my_run_2024-01-01/ optimizer.learning_rate=5e-5
    
    # Resume from specific checkpoint step
    python resume_training.py outputs/my_run_2024-01-01/ --step 50000
    
    # Combine step selection with overrides
    python resume_training.py outputs/my_run_2024-01-01/ --step 50000 training.max_steps=200000

Notes:
    - Requires .hydra/ directory in run_dir (created by start_training.py)
    - For runs without .hydra/, use start_training.py with explicit config
    - Original overrides from .hydra/overrides.yaml are applied first
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

from hydra import compose, initialize_config_dir
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
        help='Path to run directory containing .hydra/ and checkpoints'
    )
    parser.add_argument(
        '--step', 
        type=str, 
        default='latest',
        help='Checkpoint step to resume from (int or "latest", default: latest)'
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
    
    if not os.path.exists(run_dir):
        print(f"[ERROR] Run directory not found: {run_dir}")
        sys.exit(1)
    
    if not os.path.exists(hydra_dir):
        print(f"[ERROR] {hydra_dir} not found.")
        print("   Cannot resume without .hydra/ config directory.")
        print("   For runs without .hydra/, use start_training.py with explicit config.")
        sys.exit(1)
    
    # Load original overrides (if any)
    overrides_file = os.path.join(hydra_dir, 'overrides.yaml')
    if os.path.exists(overrides_file):
        original_overrides = OmegaConf.load(overrides_file)
        if original_overrides is None:
            original_overrides = []
        else:
            # Convert to list if it's a ListConfig
            original_overrides = list(original_overrides)
    else:
        original_overrides = []
    
    # Sanitize overrides: convert '+key=' to '++key=' since we're re-applying
    # to an already-composed config (where the keys may already exist)
    def sanitize_override(override: str) -> str:
        if override.startswith('+') and not override.startswith('++'):
            return '+' + override  # Convert +key= to ++key=
        return override
    
    original_overrides = [sanitize_override(o) for o in original_overrides]
    cli_overrides = [sanitize_override(o) for o in args.overrides]
    
    # Combine overrides: original + CLI (CLI takes precedence)
    all_overrides = original_overrides + cli_overrides
    
    print(f"[Resuming from: {run_dir}]")
    print(f"   Config: {hydra_dir}/config.yaml")
    print(f"   Checkpoint step: {args.step}")
    if original_overrides:
        print(f"   Original overrides: {original_overrides}")
    if cli_overrides:
        print(f"   CLI overrides: {cli_overrides}")
    
    # Initialize Hydra with the checkpoint's config directory
    # Clear any existing Hydra instance first
    GlobalHydra.instance().clear()
    
    # Use context manager for proper cleanup
    with initialize_config_dir(version_base=None, config_dir=hydra_dir, job_name="resume"):
        cfg = compose(config_name="config", overrides=all_overrides)
    
    # Import and run training
    # Import here to avoid circular imports and ensure Hydra is initialized first
    from src.utils.train_utils import setup_config_aliases, setup_seeds, setup_and_resume_training
    
    # Set up config aliases (same as start_training.py)
    cfg = setup_config_aliases(cfg)
    
    # Set seeds for reproducibility
    setup_seeds(cfg)
    
    # Run resumed training
    setup_and_resume_training(cfg, run_dir=run_dir, resume_step=args.step)
    
    print(f"\n✓ Training resumed and completed successfully")


if __name__ == '__main__':
    main()

