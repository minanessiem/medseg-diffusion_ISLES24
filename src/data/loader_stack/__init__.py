"""
Internal loader-stack package for phased data-loader refactor.

Phase 1 introduces this package as scaffolding only.
Runtime behavior remains sourced from `src.data.loaders`.
"""

from src.data.loader_stack.contracts import (
    SUPPORTED_LOADER_MODES,
    validate_supported_loader_mode,
)
from src.data.loader_stack.core import _build_loader_kwargs, _is_set
from src.data.loader_stack.factory import (
    LoaderResolution,
    resolve_dataset_identity,
    resolve_loader_contract,
)
from src.data.loader_stack.registry import (
    DatasetCapabilities,
    get_dataset_capabilities,
)

__all__ = [
    "SUPPORTED_LOADER_MODES",
    "validate_supported_loader_mode",
    "_build_loader_kwargs",
    "_is_set",
    "LoaderResolution",
    "resolve_dataset_identity",
    "resolve_loader_contract",
    "DatasetCapabilities",
    "get_dataset_capabilities",
]
