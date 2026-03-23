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
from src.utils.distribution_utils import (
    is_main_process,
    get_distribution_state,
    resolve_process_device,
    resolve_strategy,
)


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
    strategy = resolve_strategy(cfg)
    device = resolve_process_device(cfg.device, strategy=strategy)
    dist_state = get_distribution_state(strategy)
    print(
        f"Using device: {device} "
        f"(strategy={strategy}, rank={dist_state.rank}, "
        f"local_rank={dist_state.local_rank}, world_size={dist_state.world_size})"
    )
    return device


def apply_environment_runtime_context(cfg: DictConfig) -> DictConfig:
    """
    Apply environment-provided runtime context for training bootstrap.
    
    Data contract fields are read directly from dataset/data_mode/data_io/data_runtime.
    This function only maps execution context values that are intentionally sourced
    from the environment group (training paths + device).
    
    Args:
        cfg: Hydra config object
    
    Returns:
        Modified config with runtime context applied
    """
    OmegaConf.set_struct(cfg, False)
    
    # Environment runtime context (training paths)
    cfg.training.output_root = cfg.environment.training.output_root
    cfg.training.model_save_dir = cfg.environment.training.model_save_dir
    cfg.training.multi_gpu = cfg.environment.training.multi_gpu
    
    # Environment runtime context (top-level only)
    cfg.device = cfg.environment.device
    
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
    # Normalize output_root to ensure trailing slash (defensive against user input)
    output_root = cfg.environment.training.output_root.rstrip('/') + '/'
    run_output_dir = f"{output_root}{run_name}/"
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
    
    main_process = is_main_process()
    if mode == 'train':
        log_dir = f"{run_dir}tensorboard/"
        writer = SummaryWriter(log_dir=log_dir) if main_process else None
    else:
        # In evaluate mode, still create a writer for consistency (logs under runs/eval-<timestamp>)
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = f"runs/eval_{timestamp}/"
        writer = SummaryWriter(log_dir=log_dir) if main_process else None
    
    enabled_outputs = list(cfg.logging.outputs) if main_process else []
    logger = Logger(
        log_dir=log_dir,
        enabled_outputs=enabled_outputs,
        log_interval=int(cfg.logging.interval),
        table_format=cfg.logging.table_format,
        writer=writer,
        cfg=cfg.logging,  # Pass logging config
    )
    if main_process:
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

    # Keep model channels aligned with the active data contract before model build.
    sync_model_image_channels_with_data_contract(cfg)
    
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


def _resolve_expected_image_channels(cfg: DictConfig) -> int:
    """Resolve expected model image channels from the active data contract."""
    loader_mode = OmegaConf.select(cfg, "data_mode.loader_mode")
    num_modalities = OmegaConf.select(cfg, "dataset.num_modalities")
    if num_modalities is None:
        modalities = OmegaConf.select(cfg, "dataset.modalities")
        if modalities is None:
            raise ValueError(
                "Missing dataset channel contract: expected dataset.num_modalities "
                "or dataset.modalities."
            )
        num_modalities = len(modalities)
    num_modalities = int(num_modalities)

    if loader_mode == "nnunet_slices_2d":
        per_side_context_slices = int(
            OmegaConf.select(cfg, "data_mode.per_side_context_slices", default=0) or 0
        )
        num_effective_slices = (2 * per_side_context_slices) + 1
        return num_modalities * num_effective_slices

    return num_modalities


def sync_model_image_channels_with_data_contract(cfg: DictConfig) -> int:
    """Auto-sync model.image_channels from the active data contract."""
    expected_channels = _resolve_expected_image_channels(cfg)
    current_channels = OmegaConf.select(cfg, "model.image_channels")
    previous = None if current_channels is None else int(current_channels)

    if previous != expected_channels:
        OmegaConf.set_struct(cfg, False)
        OmegaConf.update(cfg, "model.image_channels", int(expected_channels), merge=False)
        OmegaConf.set_struct(cfg, True)
        if previous is None:
            print(
                "[Data Contract] Set model.image_channels from data contract: "
                f"{expected_channels}."
            )
        else:
            print(
                "[Data Contract] Updated model.image_channels to match data contract: "
                f"{previous} -> {expected_channels}."
            )

    return int(expected_channels)


def validate_model_channel_contract(cfg: DictConfig) -> None:
    """Fail fast when model input channels disagree with data contract."""
    expected_channels = _resolve_expected_image_channels(cfg)
    configured_channels = int(OmegaConf.select(cfg, "model.image_channels"))
    if configured_channels != expected_channels:
        loader_mode = OmegaConf.select(cfg, "data_mode.loader_mode")
        per_side_context_slices = int(
            OmegaConf.select(cfg, "data_mode.per_side_context_slices", default=0) or 0
        )
        raise ValueError(
            "Model image channel contract mismatch. "
            f"Expected model.image_channels={expected_channels} from data contract "
            f"(loader_mode={loader_mode}, per_side_context_slices={per_side_context_slices}), "
            f"got model.image_channels={configured_channels}."
        )


# =============================================================================
# High-Level Training Entry Points
# =============================================================================

def setup_and_start_training(cfg: DictConfig, run_dir: str):
    """
    Complete setup and training for a fresh run.
    
    Called by start_training.py after Hydra config loading.
    Handles all component initialization and runs training.
    
    Args:
        cfg: Hydra config (runtime context already applied)
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

    # Sync before logger prints resolved config for accurate channel visibility.
    sync_model_image_channels_with_data_contract(cfg)
    
    _debug("Setting up logger...")
    logger, writer = setup_logger(cfg, run_dir, mode='train')
    _debug("Logger done")
    
    _debug("Building model and diffusion...")
    diffusion = build_model_and_diffusion(cfg, device)
    _debug("Model and diffusion built")
    
    _debug("Getting dataloaders...")
    dataloaders = get_dataloaders(cfg)
    _debug(f"Dataloaders ready: {list(dataloaders.keys())}")
    
    # Contract validation: model channels must match active data contract.
    validate_model_channel_contract(cfg)
    
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


def setup_and_resume_training(cfg: DictConfig, run_dir: str, resume_step: str = 'latest', legacy_mode: bool = False):
    """
    Complete setup and training for a resumed run.
    
    Called by resume_training.py after Compose API config loading.
    Loads checkpoint and restores all training state before continuing.
    
    Args:
        cfg: Hydra config (runtime context already applied)
        run_dir: Path to run directory to resume from (with trailing slash)
        resume_step: Step to resume from (int as string or 'latest')
        legacy_mode: If True, this is a legacy run without .hydra/ or training_state.
                     EMA will be initialized fresh, best metrics reset, scheduler reconstructed.
    """
    from src.data.loaders import get_dataloaders
    from src.training.trainer import get_optimizer_and_scheduler, step_based_train
    from src.training.checkpoint_utils import load_checkpoint, load_model_state_dict_compat
    
    device = setup_device(cfg)
    # Sync before logger prints resolved config for accurate channel visibility.
    sync_model_image_channels_with_data_contract(cfg)
    
    # Load checkpoint first to get step info
    resume_state = load_checkpoint(run_dir, resume_step, cfg, device)
    print(f"✓ Loaded checkpoint from step {resume_state['global_step']}")
    
    logger, writer = setup_logger(cfg, run_dir, mode='train')
    diffusion = build_model_and_diffusion(cfg, device)
    
    # Load model weights
    missing_keys, unexpected_keys = load_model_state_dict_compat(
        diffusion, resume_state['model_state_dict']
    )
    if missing_keys or unexpected_keys:
        print(
            f"  ├─ Model weights loaded with compatibility mode "
            f"(missing={len(missing_keys)}, unexpected={len(unexpected_keys)})"
        )
    else:
        print(f"  ├─ Model weights loaded")
    
    # Get dataloaders
    dataloaders = get_dataloaders(cfg)
    
    # Contract validation: model channels must match active data contract.
    validate_model_channel_contract(cfg)
    
    optimizer, scheduler = get_optimizer_and_scheduler(cfg, diffusion)
    
    # Load optimizer state (if available)
    if resume_state.get('optimizer_state_dict'):
        optimizer.load_state_dict(resume_state['optimizer_state_dict'])
        print(f"  ├─ Optimizer state loaded")
    else:
        print(f"  ├─ Optimizer state not found (starting fresh)")
    
    # Handle training state (EMA, best metrics, scheduler)
    training_state = resume_state.get('training_state', {})
    
    if legacy_mode or not training_state:
        # Legacy mode: no training_state file, initialize everything fresh
        print(f"  ├─ [LEGACY] No training state found - initializing fresh:")
        print(f"  │   ├─ EMA: will initialize from current model weights")
        
        # Set best metric based on metric mode
        if cfg.training.checkpoint_best.enabled:
            metric_mode = cfg.training.checkpoint_best.get('metric_mode', 'max')
            if metric_mode == 'max':
                print(f"  │   ├─ Best metric: -inf (mode=max, first validation will be saved)")
            else:
                print(f"  │   ├─ Best metric: inf (mode=min, first validation will be saved)")
        
        # Reconstruct scheduler to resume step
        if scheduler:
            global_step = resume_state['global_step']
            sched_cfg = cfg.scheduler
            if sched_cfg.get('step_frequency', 'per_step') == 'per_step':
                print(f"  │   └─ Scheduler: fast-forwarding {global_step} steps...")
                for _ in range(global_step):
                    scheduler.step()
                print(f"  │       └─ Scheduler reconstructed to step {global_step}")
            else:
                print(f"  │   └─ Scheduler: per-interval mode, will adjust naturally")
        
        # Clear training_state so trainer initializes fresh
        resume_state['training_state'] = {}
    else:
        # Standard mode: load scheduler state if available
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
