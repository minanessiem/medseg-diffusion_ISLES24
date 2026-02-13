"""
Distribution runtime utilities.

This module centralizes strategy parsing and PyTorch distributed runtime helpers
for DP/DDP execution modes. It is intentionally framework-adjacent and contains
no training-loop or logging responsibilities.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from functools import wraps
from typing import Any, Callable, Optional

import torch
import torch.distributed as dist


SUPPORTED_STRATEGIES = {"dp", "ddp"}


@dataclass(frozen=True)
class DistributionState:
    """Runtime distribution metadata for the current process."""

    strategy: str
    rank: int
    local_rank: int
    world_size: int


def resolve_strategy(cfg: Any) -> str:
    """
    Resolve and validate distribution strategy from config.

    Defaults to ``dp`` when strategy is absent to preserve legacy behavior.
    """
    strategy = "dp"

    distribution_cfg = getattr(cfg, "distribution", None)
    if distribution_cfg is not None:
        strategy = str(getattr(distribution_cfg, "strategy", strategy)).strip().lower()

    if strategy not in SUPPORTED_STRATEGIES:
        allowed = ", ".join(sorted(SUPPORTED_STRATEGIES))
        raise ValueError(f"Unsupported distribution strategy '{strategy}'. Expected one of: {allowed}")

    return strategy


def get_rank() -> int:
    """Read global rank from environment (defaults to 0)."""
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    """Read process-local rank from environment (defaults to 0)."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    """Read world size from environment (defaults to 1)."""
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_distribution_state(strategy: str) -> DistributionState:
    """Build a snapshot of rank metadata for the current process."""
    if strategy == "ddp":
        return DistributionState(
            strategy=strategy,
            rank=get_rank(),
            local_rank=get_local_rank(),
            world_size=get_world_size(),
        )
    return DistributionState(strategy=strategy, rank=0, local_rank=0, world_size=1)


def validate_ddp_runtime() -> None:
    """Fail fast when required DDP runtime prerequisites are invalid."""
    required_env = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [k for k in required_env if k not in os.environ]
    if missing:
        missing_csv = ", ".join(missing)
        raise RuntimeError(
            "DDP strategy requires torchrun-provided environment variables. "
            f"Missing: {missing_csv}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("DDP strategy currently requires CUDA, but CUDA is not available.")

    world_size = get_world_size()
    rank = get_rank()
    local_rank = get_local_rank()

    if world_size < 1:
        raise RuntimeError(f"Invalid WORLD_SIZE={world_size}. Expected WORLD_SIZE >= 1.")
    if rank < 0 or rank >= world_size:
        raise RuntimeError(f"Invalid RANK={rank} for WORLD_SIZE={world_size}.")
    if local_rank < 0:
        raise RuntimeError(f"Invalid LOCAL_RANK={local_rank}. Expected LOCAL_RANK >= 0.")

    visible_cuda = torch.cuda.device_count()
    if local_rank >= visible_cuda:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {visible_cuda} CUDA device(s) visible."
        )


def resolve_process_device(device_value: str, strategy: str) -> torch.device:
    """
    Resolve process device for the selected strategy.

    For DDP, each process is pinned to ``cuda:LOCAL_RANK``.
    For DP, behavior mirrors the current single-process setup.
    """
    if strategy == "ddp":
        validate_ddp_runtime()
        local_rank = get_local_rank()
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")

    if device_value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_value)


def init_process_group_if_needed(
    strategy: str,
    backend: str = "nccl",
    timeout_minutes: int = 30,
) -> DistributionState:
    """
    Initialize torch.distributed process group when strategy is DDP.

    Returns the current process distribution state for convenience.
    """
    if strategy != "ddp":
        return get_distribution_state(strategy)

    validate_ddp_runtime()
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build.")

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=timeout_minutes),
        )

    return DistributionState(
        strategy="ddp",
        rank=dist.get_rank(),
        local_rank=get_local_rank(),
        world_size=dist.get_world_size(),
    )


def destroy_process_group_if_needed() -> None:
    """Destroy torch.distributed process group if initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_distributed() -> bool:
    """Return True when torch.distributed is active with more than one process."""
    return bool(
        dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )


def is_main_process() -> bool:
    """Return True for rank 0, or True in non-distributed mode."""
    if is_distributed():
        return dist.get_rank() == 0
    return get_rank() == 0


def barrier_if_needed() -> None:
    """Synchronize processes when distributed runtime is active."""
    if is_distributed():
        dist.barrier()


def rank_zero_only(func: Callable[..., Any]) -> Callable[..., Optional[Any]]:
    """Execute wrapped function only on main process; no-op on non-zero ranks."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Optional[Any]:
        if not is_main_process():
            return None
        return func(*args, **kwargs)

    return wrapper


