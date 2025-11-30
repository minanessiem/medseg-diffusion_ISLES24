"""
Training utilities shared between start_training.py and resume_training.py.

This module contains setup functions that are common to both fresh and resumed
training runs, promoting code reuse and consistency.
"""

import math
import os
import random
import copy

import numpy as np
import torch
from typing import Iterable, List, Tuple, Optional
from omegaconf import OmegaConf, DictConfig, ListConfig
from torch.utils.tensorboard import SummaryWriter


# =============================================================================
# Gradient and Parameter Norm Utilities (existing functions)
# =============================================================================

def calc_grad_norm(params: Iterable[torch.nn.Parameter]) -> float:
    """L2 norm of gradients (only params with .grad)."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return math.sqrt(total)


def calc_param_norm(params: Iterable[torch.nn.Parameter]) -> float:
    """L2 norm of parameters."""
    total = sum(p.data.norm(2).item() ** 2 for p in params)
    return math.sqrt(total)


# =============================================================================
# Multi-GPU Utilities
# =============================================================================

def _parse_multi_gpu_flag(flag):
    """
    Parse multi-GPU configuration flag.
    
    Args:
        flag: Can be None, False, "false", "none", "", a list of GPU ids,
              a tuple of GPU ids, ListConfig, or comma-separated string.
    
    Returns:
        List of GPU ids or None if multi-GPU is disabled.
    
    Examples:
        >>> _parse_multi_gpu_flag(None)
        None
        >>> _parse_multi_gpu_flag([0, 1])
        [0, 1]
        >>> _parse_multi_gpu_flag("0,1,2")
        [0, 1, 2]
    """
    if not flag or str(flag).lower() in {"false", "none", ""}:
        return None
    # Handle OmegaConf ListConfig, plain list, or tuple
    if isinstance(flag, (list, tuple, ListConfig)):
        return [int(x) for x in flag]
    return [int(x.strip()) for x in str(flag).split(',') if x.strip()]


# =============================================================================
# Setup Functions
# =============================================================================

def setup_seeds(cfg: DictConfig) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        cfg: Config with random_seed key
    """
    torch.manual_seed(cfg.random_seed)
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)


def setup_device(cfg: DictConfig) -> torch.device:
    """
    Configure and return the training device.
    
    Args:
        cfg: Config with device key ('auto', 'cuda', 'cpu', etc.)
    
    Returns:
        torch.device configured for training
    """
    # Device configuration (auto-detect with cfg override)
    if cfg.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(cfg.device)
    print(f'Using device: {device}')
    return device


def setup_config_aliases(cfg: DictConfig) -> DictConfig:
    """
    Set up config aliases for backwards compatibility.
    
    Copies environment-specific settings to their expected locations
    in the config tree for components that expect them there.
    
    Args:
        cfg: Hydra config object
    
    Returns:
        Modified config with aliases set
    """
    OmegaConf.set_struct(cfg, False)
    
    # Environment aliases (training paths)
    cfg.training.output_root = cfg.environment.training.output_root
    cfg.training.model_save_dir = cfg.environment.training.model_save_dir
    cfg.training.multi_gpu = cfg.environment.training.multi_gpu
    
    # Environment aliases (top-level and dataset)
    cfg.device = cfg.environment.device
    cfg.dataset.dir = cfg.environment.dataset.dir
    cfg.dataset.json_list = cfg.environment.dataset.json_list
    cfg.dataset.num_train_workers = cfg.environment.dataset.num_train_workers
    cfg.dataset.num_valid_workers = cfg.environment.dataset.num_valid_workers
    cfg.dataset.num_test_workers = cfg.environment.dataset.num_test_workers
    cfg.dataset.train_batch_size = cfg.environment.dataset.train_batch_size
    cfg.dataset.test_batch_size = cfg.environment.dataset.test_batch_size
    
    OmegaConf.set_struct(cfg, True)
    return cfg


def setup_output_directory(cfg: DictConfig, run_name: str) -> str:
    """
    Create output directory structure for a new run.
    
    Creates:
        - {output_root}/{run_name}/
        - {output_root}/{run_name}/tensorboard/
        - {output_root}/{run_name}/{model_save_dir}/
    
    Args:
        cfg: Config with environment.training settings
        run_name: Name of the run (used as subdirectory)
    
    Returns:
        Path to run output directory (with trailing slash)
    """
    run_output_dir = f"{cfg.environment.training.output_root}{run_name}/"
    os.makedirs(run_output_dir, exist_ok=True)
    os.makedirs(f"{run_output_dir}tensorboard/", exist_ok=True)
    os.makedirs(f"{run_output_dir}{cfg.environment.training.model_save_dir}", exist_ok=True)
    return run_output_dir


def setup_logger(
    cfg: DictConfig, 
    run_dir: str, 
    mode: str = 'train',
    timestamp: Optional[str] = None
) -> Tuple['Logger', SummaryWriter]:
    """
    Initialize tensorboard writer and logger.
    
    Args:
        cfg: Config with logging settings
        run_dir: Path to run directory (with trailing slash)
        mode: 'train' or 'evaluate'
        timestamp: Timestamp string for eval mode log naming
    
    Returns:
        Tuple of (Logger instance, SummaryWriter instance)
    """
    from src.utils.logger import Logger
    from datetime import datetime
    
    if mode == 'train':
        log_dir = f"{run_dir}tensorboard/"
        writer = SummaryWriter(log_dir=log_dir)
    else:
        # In evaluate mode, still create a writer for consistency (logs under runs/eval-<timestamp>)
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = f"runs/eval_{timestamp}/"
        writer = SummaryWriter(log_dir=log_dir)
    
    logger = Logger(
        log_dir=log_dir,
        enabled_outputs=list(cfg.logging.outputs),
        log_interval=int(cfg.logging.interval),
        table_format=cfg.logging.table_format,
        writer=writer,
        cfg=cfg.logging,  # Pass logging config
    )
    logger.print_config(OmegaConf.to_yaml(cfg, resolve=True))
    
    return logger, writer


def build_model_and_diffusion(cfg: DictConfig, device: torch.device):
    """
    Build model and diffusion, handling multi-GPU if configured.
    
    Multi-GPU: Wraps UNet BEFORE building diffusion to ensure
    gradients flow correctly to all GPUs, especially for OpenAI adapter.
    
    Args:
        cfg: Config with model, diffusion, and multi_gpu settings
        device: Target device for model
    
    Returns:
        Diffusion model (potentially with DataParallel-wrapped UNet)
    
    Raises:
        ValueError: If requested GPU id exceeds visible GPUs
    """
    from src.models import build_model
    from src.diffusion.diffusion import Diffusion
    
    # Build model
    unet = build_model(cfg).to(device)
    
    # Multi-GPU: Wrap UNet BEFORE building diffusion
    # This ensures gradients flow correctly to all GPUs, especially for OpenAI adapter
    gpu_ids = _parse_multi_gpu_flag(cfg.environment.training.multi_gpu)
    if gpu_ids:
        visible = torch.cuda.device_count()
        if max(gpu_ids) >= visible:
            raise ValueError(f"Requested GPU id {max(gpu_ids)} but only {visible} GPUs visible.")
        print(f"Using GPUs {gpu_ids} with torch.nn.DataParallel")
        unet = torch.nn.DataParallel(unet, device_ids=gpu_ids).cuda(gpu_ids[0])
        print(f"  Wrapped UNet in DataParallel (primary device: cuda:{gpu_ids[0]})")
    
    # Build diffusion with potentially wrapped UNet
    diffusion = Diffusion.build_diffusion(unet, cfg, device)
    
    # Ensure diffusion is on correct device
    if not gpu_ids:
        diffusion = diffusion.to(device)
    else:
        # Already on correct device via UNet wrapping
        print(f"  Diffusion built with multi-GPU UNet")
    
    return diffusion


# =============================================================================
# High-Level Training Entry Points
# =============================================================================

def setup_and_start_training(cfg: DictConfig, run_dir: str):
    """
    Complete setup and training for a fresh run.
    
    Called by start_training.py after Hydra config loading.
    Handles all component initialization and runs training.
    
    Args:
        cfg: Hydra config (with aliases already set up)
        run_dir: Path to run output directory (with trailing slash)
    """
    import sys
    debug = cfg.get("debug", False)
    
    def _debug(msg):
        if debug:
            print(f"[DEBUG:setup_and_start_training] {msg}", flush=True)
            sys.stdout.flush()
    
    _debug("ENTER: setup_and_start_training()")
    
    # Ultra-granular import debugging
    _debug("Step A: About to import src.data.loaders...")
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        from src.data.loaders import get_dataloaders
        _debug("Step A: src.data.loaders imported successfully")
    except Exception as e:
        _debug(f"Step A: FAILED to import src.data.loaders: {e}")
        raise
    
    _debug("Step B: About to import src.training.trainer...")
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        from src.training.trainer import get_optimizer_and_scheduler, step_based_train
        _debug("Step B: src.training.trainer imported successfully")
    except Exception as e:
        _debug(f"Step B: FAILED to import src.training.trainer: {e}")
        raise
    
    _debug("Setting up device...")
    device = setup_device(cfg)
    _debug(f"Device: {device}")
    
    _debug("Setting up logger...")
    logger, writer = setup_logger(cfg, run_dir, mode='train')
    _debug("Logger done")
    
    _debug("Building model and diffusion...")
    diffusion = build_model_and_diffusion(cfg, device)
    _debug("Model and diffusion built")
    
    # Get dataloaders
    _debug("Getting dataloaders...")
    dataloaders = get_dataloaders(cfg)
    _debug(f"Dataloaders ready: {list(dataloaders.keys())}")
    
    # Dataset-specific validation
    if cfg.dataset.name == 'isles24':
        assert cfg.model.image_channels == len(cfg.dataset.modalities), \
            "Model image_channels must match number of modalities"
    
    _debug("Getting optimizer and scheduler...")
    optimizer, scheduler = get_optimizer_and_scheduler(cfg, diffusion)
    _debug("Optimizer and scheduler ready")
    
    # Run step-based training
    _debug("Starting step_based_train()...")
    step_based_train(
        cfg,
        diffusion,
        dataloaders,
        optimizer,
        scheduler,
        logger,
        run_dir=run_dir,
        resume_state=None  # Fresh run
    )
    _debug("step_based_train() complete")
    
    logger.close()
    _debug("EXIT: setup_and_start_training()")


def setup_and_resume_training(cfg: DictConfig, run_dir: str, resume_step: str = 'latest'):
    """
    Complete setup and training for a resumed run.
    
    Called by resume_training.py after Compose API config loading.
    Loads checkpoint and restores all training state before continuing.
    
    Args:
        cfg: Hydra config (with aliases already set up)
        run_dir: Path to run directory to resume from (with trailing slash)
        resume_step: Step to resume from (int as string or 'latest')
    """
    from src.data.loaders import get_dataloaders
    from src.training.trainer import get_optimizer_and_scheduler, step_based_train
    from src.training.checkpoint_utils import load_checkpoint
    
    device = setup_device(cfg)
    
    # Load checkpoint first to get step info
    resume_state = load_checkpoint(run_dir, resume_step, cfg, device)
    print(f"✓ Loaded checkpoint from step {resume_state['global_step']}")
    
    logger, writer = setup_logger(cfg, run_dir, mode='train')
    diffusion = build_model_and_diffusion(cfg, device)
    
    # Load model weights
    diffusion.load_state_dict(resume_state['model_state_dict'])
    print(f"  ├─ Model weights loaded")
    
    # Get dataloaders
    dataloaders = get_dataloaders(cfg)
    
    # Dataset-specific validation
    if cfg.dataset.name == 'isles24':
        assert cfg.model.image_channels == len(cfg.dataset.modalities), \
            "Model image_channels must match number of modalities"
    
    optimizer, scheduler = get_optimizer_and_scheduler(cfg, diffusion)
    
    # Load optimizer state (if available)
    if resume_state.get('optimizer_state_dict'):
        optimizer.load_state_dict(resume_state['optimizer_state_dict'])
        print(f"  ├─ Optimizer state loaded")
    
    # Load scheduler state (if available)
    training_state = resume_state.get('training_state', {})
    if scheduler and training_state.get('scheduler_state_dict'):
        scheduler.load_state_dict(training_state['scheduler_state_dict'])
        print(f"  ├─ Scheduler state loaded")
    
    print(f"  └─ Resuming training from step {resume_state['global_step']}")
    
    # Run step-based training with resume state
    step_based_train(
        cfg,
        diffusion,
        dataloaders,
        optimizer,
        scheduler,
        logger,
        run_dir=run_dir,
        resume_state=resume_state
    )
    
    logger.close()
