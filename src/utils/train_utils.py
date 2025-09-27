import math
import torch
from typing import Iterable

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