"""
Factory for building optimizers and schedulers from config.

Separates concerns:
- Optimizer: Update rule (Adam, AdamW)
- Scheduler: LR schedule (warmup_cosine, etc.)
- Training: Gradient techniques (clipping, etc.)
"""

import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from omegaconf import DictConfig

from .schedulers import WarmupCosineSchedule, WarmupConstantSchedule


def build_optimizer(model, cfg: DictConfig):
    """
    Build optimizer from config.
    
    Args:
        model: Model to optimize
        cfg: Config with cfg.optimizer defined
    
    Returns:
        torch.optim.Optimizer instance
    
    Raises:
        ValueError: If optimizer_class not supported
        KeyError: If required config key missing
    
    Example:
        >>> optimizer = build_optimizer(model, cfg)
    """
    opt_cfg = cfg.optimizer
    optimizer_class = opt_cfg.optimizer_class.lower()
    
    # Required parameters - will raise KeyError if missing
    lr = opt_cfg.learning_rate
    betas = tuple(opt_cfg.betas)
    eps = opt_cfg.eps
    weight_decay = opt_cfg.weight_decay
    amsgrad = opt_cfg.amsgrad
    
    if optimizer_class == 'adam':
        optimizer = Adam(
            model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
    
    elif optimizer_class == 'adamw':
        optimizer = AdamW(
            model.parameters(),
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
    
    else:
        raise ValueError(
            f"Unknown optimizer_class: '{optimizer_class}'. "
            f"Supported: adam, adamw"
        )
    
    print(f"Built optimizer: {optimizer_class.upper()} (lr={lr:.2e}, weight_decay={weight_decay})")
    return optimizer


def build_scheduler(optimizer, cfg: DictConfig):
    """
    Build LR scheduler from config.
    
    Args:
        optimizer: Optimizer to schedule
        cfg: Config with cfg.scheduler and cfg.training defined
    
    Returns:
        torch.optim.lr_scheduler instance or None
    
    Raises:
        ValueError: If scheduler_type not supported
        KeyError: If required config key missing
    
    Example:
        >>> scheduler = build_scheduler(optimizer, cfg)
    """
    sched_cfg = cfg.scheduler
    scheduler_type = sched_cfg.scheduler_type.lower()
    max_steps = cfg.training.max_steps
    
    if scheduler_type == 'warmup_cosine':
        # Compute warmup_steps
        warmup_steps = sched_cfg.warmup_steps
        if warmup_steps is None:
            warmup_fraction = sched_cfg.warmup_fraction
            warmup_steps = int(max_steps * warmup_fraction)
        
        eta_min = sched_cfg.eta_min
        
        scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=max_steps,
            eta_min=eta_min
        )
        print(f"Built scheduler: WarmupCosineSchedule")
        print(f"  Warmup: {warmup_steps} steps ({warmup_steps/max_steps*100:.1f}%)")
        print(f"  Cosine: {max_steps - warmup_steps} steps (LR: base → {eta_min})")
    
    elif scheduler_type == 'warmup_constant':
        warmup_steps = sched_cfg.warmup_steps
        if warmup_steps is None:
            warmup_fraction = sched_cfg.warmup_fraction
            warmup_steps = int(max_steps * warmup_fraction)
        
        scheduler = WarmupConstantSchedule(
            optimizer,
            warmup_steps=warmup_steps
        )
        print(f"Built scheduler: WarmupConstantSchedule")
        print(f"  Warmup: {warmup_steps} steps, then constant")
    
    elif scheduler_type == 'cosine':
        eta_min = sched_cfg.eta_min
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max_steps,
            eta_min=eta_min
        )
        print(f"Built scheduler: CosineAnnealingLR (no warmup)")
    
    elif scheduler_type == 'reduce_lr':
        factor = sched_cfg.factor
        patience = sched_cfg.patience
        threshold = sched_cfg.threshold
        cooldown = sched_cfg.cooldown
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            threshold=threshold,
            cooldown=cooldown
        )
        print(f"Built scheduler: ReduceLROnPlateau")
        print(f"  Factor={factor}, Patience={patience}")
        print(f"  WARNING: Consider warmup_cosine for diffusion models")
    
    elif scheduler_type == 'constant':
        scheduler = None  # No scheduling
        print("Built scheduler: None (constant LR)")
    
    else:
        raise ValueError(
            f"Unknown scheduler_type: '{scheduler_type}'. "
            f"Supported: warmup_cosine, warmup_constant, cosine, reduce_lr, constant"
        )
    
    return scheduler

