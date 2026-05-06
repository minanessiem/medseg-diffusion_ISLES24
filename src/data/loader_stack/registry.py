"""
Dataset capability registry for loader routing.

Phase 4 note:
- Used by the loader factory to resolve dataset capability metadata.
- `implementation_state` gates whether a dataset route is active at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.data.loader_stack.contracts import SUPPORTED_LOADER_MODES


@dataclass(frozen=True)
class DatasetCapabilities:
    """
    Static dataset capability descriptor used by future loader routing.
    """

    dataset_id: str
    supported_loader_modes: tuple[str, ...]
    loader_module: str
    implementation_state: str


DEFAULT_DATASET_REGISTRY: dict[str, DatasetCapabilities] = {
    # ISLES24 currently covers all active loader modes.
    "isles24": DatasetCapabilities(
        dataset_id="isles24",
        supported_loader_modes=tuple(SUPPORTED_LOADER_MODES),
        loader_module="src.data.loader_stack.isles24_loader",
        implementation_state="legacy-runtime",
    ),
    # ISLES26 starts with online/full-volume paths first.
    "isles26": DatasetCapabilities(
        dataset_id="isles26",
        supported_loader_modes=(
            "online_slices_3d_to_2d",
            "full_volumes_3d",
        ),
        loader_module="src.data.loader_stack.isles26_loader",
        implementation_state="online-runtime",
    ),
}


def get_dataset_capabilities(dataset_id: str) -> DatasetCapabilities:
    """
    Resolve dataset capabilities from the default registry.
    """
    key = str(dataset_id).strip().lower()
    if key not in DEFAULT_DATASET_REGISTRY:
        supported = ", ".join(sorted(DEFAULT_DATASET_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported dataset id '{dataset_id}'. Supported dataset ids: [{supported}]"
        )
    return DEFAULT_DATASET_REGISTRY[key]


def phase_marker() -> str:
    """
    Return a stable marker string for incremental refactor checks.
    """
    return "loader_stack.registry.phase4_5"
