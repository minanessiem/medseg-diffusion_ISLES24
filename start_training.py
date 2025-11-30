#!/usr/bin/env python3
"""
Start a new training run.

This is the main entry point for fresh training runs. It uses Hydra's
@hydra.main() decorator to load configuration from the configs/ directory.

For resuming an interrupted run, use resume_training.py instead.

Usage:
    python start_training.py --config-name <config>
    
Examples:
    # Local training with default config
    python start_training.py --config-name local
    
    # Cluster training with DDIM 250 steps
    python start_training.py --config-name cluster_ddim250_multitask
    
    # Override specific settings
    python start_training.py --config-name local training.max_steps=50000
    
    # Enable debug logging
    python start_training.py --config-name local debug=true
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import shutil
import os
import sys
from datetime import datetime

from src.utils.run_name import generate_run_name
from src.utils.train_utils import (
    setup_seeds,
    setup_config_aliases,
    setup_output_directory,
    setup_and_start_training,
    setup_device,
    setup_logger,
    build_model_and_diffusion,
)

# Debug flag - set via config override: debug=true
DEBUG = False

def debug_print(msg: str):
    """Print debug message if DEBUG is enabled."""
    if DEBUG:
        print(f"[DEBUG] {msg}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    """
    Main entry point for training.
    
    Handles:
    - Config alias setup for backwards compatibility
    - Random seed initialization
    - Output directory creation
    - Moving Hydra logs to run directory
    - Training or evaluation based on cfg.mode
    """
    global DEBUG
    DEBUG = cfg.get("debug", False)
    
    debug_print("="*60)
    debug_print("START: main() entered")
    debug_print("="*60)
    
    # Set up config aliases for compatibility
    debug_print("STEP 1: Setting up config aliases...")
    cfg = setup_config_aliases(cfg)
    debug_print("STEP 1: Config aliases done")
    
    # Set seeds for reproducibility
    debug_print("STEP 2: Setting up seeds...")
    setup_seeds(cfg)
    debug_print("STEP 2: Seeds done")
    
    # Use overridden timestamp/run_name if provided
    debug_print("STEP 3: Generating timestamp and run_name...")
    timestamp = cfg.get("timestamp") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = cfg.get("run_name") or generate_run_name(cfg, timestamp)
    debug_print(f"STEP 3: run_name = {run_name}")
    
    # Create run directory structure
    debug_print("STEP 4: Creating output directory structure...")
    run_output_dir = setup_output_directory(cfg, run_name)
    debug_print(f"STEP 4: run_output_dir = {run_output_dir}")
    
    # Capture original Hydra run.dir BEFORE updating (needed for artifact moving)
    debug_print("STEP 5: Capturing original Hydra run.dir...")
    from hydra.core.hydra_config import HydraConfig
    original_hydra_dir = HydraConfig.get().run.dir
    debug_print(f"STEP 5: original_hydra_dir = {original_hydra_dir}")
    
    # Update Hydra run.dir to point to our run directory
    debug_print("STEP 6: Updating Hydra run.dir in config...")
    OmegaConf.set_struct(cfg, False)
    OmegaConf.update(cfg, "hydra.run.dir", run_output_dir)
    OmegaConf.set_struct(cfg, True)
    debug_print("STEP 6: Hydra run.dir updated")
    
    # Move early Hydra logs (e.g., main.log) from original location to run_dir
    debug_print("STEP 7: Moving Hydra artifacts...")
    _move_hydra_artifacts(original_hydra_dir, run_output_dir)
    debug_print("STEP 7: Hydra artifacts moved")
    
    debug_print(f"STEP 8: Mode = {cfg.mode}")
    
    if cfg.mode == "train":
        # Run training using shared utility
        debug_print("STEP 9: Entering setup_and_start_training()...")
        setup_and_start_training(cfg, run_output_dir)
        debug_print("STEP 9: Training complete")
        
    elif cfg.mode == "evaluate":
        # Evaluation mode: Load model and visualize
        debug_print("STEP 9: Entering _run_evaluation()...")
        _run_evaluation(cfg, run_output_dir, timestamp)
        debug_print("STEP 9: Evaluation complete")


def _move_hydra_artifacts(source_dir: str, run_output_dir: str) -> None:
    """
    Move Hydra-generated artifacts from source directory to run directory.
    
    Moves:
    - main.log (if exists)
    - .hydra/ directory (config snapshots, overrides)
    
    Args:
        source_dir: Original Hydra run.dir where artifacts were created
        run_output_dir: Target run directory
    """
    # Convert to absolute path for clarity
    temp_log_dir = os.path.abspath(source_dir)
    
    print(f"\n[Hydra Artifacts]")
    print(f"   Source (original hydra.run.dir): {source_dir} -> {temp_log_dir}")
    print(f"   Target (run output dir):         {run_output_dir}")
    
    # Check what exists in temp directory
    if os.path.exists(temp_log_dir):
        temp_contents = os.listdir(temp_log_dir)
        print(f"   Temp dir contents: {temp_contents}")
    else:
        print(f"   [WARN] Temp dir does not exist: {temp_log_dir}")
    
    # Move main.log
    early_log = f"{temp_log_dir}/main.log"
    if os.path.exists(early_log):
        os.makedirs(run_output_dir, exist_ok=True)
        target_log = f"{run_output_dir}/main.log"
        # Remove target if it exists to avoid conflict
        if os.path.exists(target_log):
            os.remove(target_log)
        shutil.move(early_log, target_log)
        print(f"   [OK] Moved main.log -> {run_output_dir}")
    else:
        print(f"   [INFO] No main.log found at {early_log}")
    
    # Move .hydra/ metadata folder to run_dir
    hydra_dir = f"{temp_log_dir}/.hydra"
    if os.path.exists(hydra_dir):
        # List .hydra contents before moving
        hydra_contents = os.listdir(hydra_dir)
        print(f"   .hydra/ contents: {hydra_contents}")
        
        target_hydra = f"{run_output_dir}/.hydra"
        # Remove target if it exists to avoid nested directories
        if os.path.exists(target_hydra):
            shutil.rmtree(target_hydra)
            print(f"   [INFO] Removed existing {target_hydra}")
        shutil.move(hydra_dir, target_hydra)
        print(f"   [OK] Moved .hydra/ -> {target_hydra}")
        
        # Verify the move
        if os.path.exists(target_hydra):
            final_contents = os.listdir(target_hydra)
            print(f"   [OK] Verified .hydra/ at destination: {final_contents}")
        else:
            print(f"   [ERROR] .hydra/ not found at destination after move!")
    else:
        print(f"   [WARN] No .hydra/ found at {hydra_dir}")
        print(f"      This may happen if Hydra's run.dir is set to '.' in config")
        
        # Check if .hydra exists in current working directory instead
        cwd_hydra = os.path.join(os.getcwd(), '.hydra')
        if os.path.exists(cwd_hydra):
            print(f"   [INFO] Found .hydra/ in CWD: {cwd_hydra}")
            hydra_contents = os.listdir(cwd_hydra)
            print(f"      Contents: {hydra_contents}")
            
            target_hydra = f"{run_output_dir}/.hydra"
            if os.path.exists(target_hydra):
                shutil.rmtree(target_hydra)
            shutil.copytree(cwd_hydra, target_hydra)
            print(f"   [OK] Copied .hydra/ from CWD -> {target_hydra}")


def _run_evaluation(cfg: DictConfig, run_output_dir: str, timestamp: str) -> None:
    """
    Run evaluation mode: load model and visualize predictions.
    
    Args:
        cfg: Hydra config
        run_output_dir: Path to output directory
        timestamp: Timestamp for logging
    """
    import torch
    from src.evaluation.evaluator import visualize_best_model_predictions
    from src.data.loaders import get_dataloaders
    
    device = setup_device(cfg)
    logger, writer = setup_logger(cfg, run_output_dir, mode='evaluate', timestamp=timestamp)
    diffusion = build_model_and_diffusion(cfg, device)
    
    # Get dataloaders (need sample_dl for visualization)
    dataloaders = get_dataloaders(cfg)
    sample_dl = dataloaders['sample']
    
    # Load checkpoint
    # Stub for evaluation: Load model/EMA from config and visualize
    # Assuming load_checkpoint function exists or add a simple loader
    checkpoint_path = f"{cfg.training.evaluation.run_dir}{cfg.training.checkpoint_path_template.format(cfg.training.evaluation.step)}"
    if cfg.training.evaluation.ema_rate is not None:
        checkpoint_path = checkpoint_path.replace(
            '.pth', 
            f'_ema_{cfg.training.evaluation.ema_rate}_{cfg.training.evaluation.step:06d}.pth'
        )
    diffusion.load_state_dict(torch.load(checkpoint_path))
    
    visualize_best_model_predictions(diffusion, sample_dl.dataset, cfg)
    logger.close()


if __name__ == '__main__':
    main()

