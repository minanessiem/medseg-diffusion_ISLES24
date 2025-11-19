"""
Learning rate schedulers for diffusion model training.

Provides custom schedulers beyond PyTorch defaults, particularly
warmup-based schedulers common in diffusion literature.
"""

import math
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineSchedule(_LRScheduler):
    """
    Linear warmup followed by cosine annealing.
    
    Based on DiffSwinTr (MICCAI 2023) and Transformer literature.
    
    Schedule:
        - Warmup: Linear 0 → base_lr over warmup_steps
        - Cosine: Cosine base_lr → eta_min over remaining steps
    
    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Steps for linear 0→base_lr warmup
        total_steps: Total training steps (for cosine period)
        eta_min: Minimum LR at end (default: 0)
        last_epoch: Starting step index (default: -1)
    
    Example:
        >>> optimizer = AdamW(model.parameters(), lr=1e-4)
        >>> scheduler = WarmupCosineSchedule(
        ...     optimizer, warmup_steps=10000, total_steps=100000
        ... )
        >>> for step in range(100000):
        ...     loss.backward()
        ...     optimizer.step()
        ...     scheduler.step()  # Call every step
    """
    
    def __init__(self, optimizer, warmup_steps, total_steps, 
                 eta_min=0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Compute learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Warmup: linear 0 → base_lr
            warmup_factor = self.last_epoch / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine: base_lr → eta_min
            progress = (self.last_epoch - self.warmup_steps) / \
                       (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.eta_min + (base_lr - self.eta_min) * cosine_decay
                for base_lr in self.base_lrs
            ]


class WarmupConstantSchedule(_LRScheduler):
    """
    Linear warmup followed by constant learning rate.
    
    Useful for ablation studies or when you don't want decay.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Steps for linear 0→base_lr warmup
        last_epoch: Starting step index (default: -1)
    
    Example:
        >>> scheduler = WarmupConstantSchedule(optimizer, warmup_steps=10000)
    """
    
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Compute learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Warmup: linear 0 → base_lr
            warmup_factor = self.last_epoch / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Constant: maintain base_lr
            return self.base_lrs

