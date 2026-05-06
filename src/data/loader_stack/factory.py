"""
Loader factory for dataset/mode dispatch.

Phase 2 note:
- Used by `src.data.loaders` compatibility facade for dataset/mode resolution.
- Runtime dispatch still targets ISLES24 classes while additional dataset loaders are implemented.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.data.loader_stack.contracts import validate_supported_loader_mode
from src.data.loader_stack.registry import DatasetCapabilities, get_dataset_capabilities


@dataclass(frozen=True)
class LoaderResolution:
    """
    Normalized dataset/mode resolution output for future factory routing.
    """

    dataset_id: str
    loader_mode: str
    capabilities: DatasetCapabilities


def resolve_dataset_identity(dataset_id: object, dataset_name: object) -> str:
    """
    Resolve canonical dataset identity from id/name fields.
    """
    candidates = [dataset_id, dataset_name]
    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).strip().lower()
        if text:
            return text
    raise ValueError(
        "Cannot resolve dataset identity from dataset.id or dataset.name."
    )


def resolve_loader_contract(
    dataset_id: object,
    dataset_name: object,
    loader_mode: object,
) -> LoaderResolution:
    """
    Resolve dataset and validate loader mode support using registry metadata.
    """
    normalized_dataset_id = resolve_dataset_identity(dataset_id, dataset_name)
    capabilities = get_dataset_capabilities(normalized_dataset_id)
    normalized_loader_mode = validate_supported_loader_mode(loader_mode)
    if normalized_loader_mode not in capabilities.supported_loader_modes:
        allowed = ", ".join(capabilities.supported_loader_modes)
        raise ValueError(
            f"dataset '{normalized_dataset_id}' does not support loader_mode "
            f"'{normalized_loader_mode}'. Supported: [{allowed}]"
        )
    return LoaderResolution(
        dataset_id=normalized_dataset_id,
        loader_mode=normalized_loader_mode,
        capabilities=capabilities,
    )


def phase_marker() -> str:
    """
    Return a stable marker string for incremental refactor checks.
    """
    return "loader_stack.factory.phase3"
