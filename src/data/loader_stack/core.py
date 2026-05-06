"""
Shared loader-stack primitives.

Phase 1 note:
- This module starts receiving low-risk shared helpers.
- Runtime wiring is still controlled by `src.data.loaders`.
"""

from __future__ import annotations


def _is_set(value):
    """
    Shared helper for checking whether a config value is effectively set.
    """
    if value is None:
        return False
    if isinstance(value, str):
        return len(value.strip()) > 0
    return True


def _build_loader_kwargs(
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
):
    """
    Build canonical DataLoader kwargs while preserving existing worker semantics.
    """
    kwargs = {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(persistent_workers) if int(num_workers) > 0 else False,
    }
    if int(num_workers) > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return kwargs


def phase_marker() -> str:
    """
    Return a stable marker string for incremental refactor checks.
    """
    return "loader_stack.core.phase1"
