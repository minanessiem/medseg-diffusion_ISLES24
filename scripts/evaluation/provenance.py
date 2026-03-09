"""
Provenance parsing helpers for evaluation sample identity.

This module centralizes strict parsing rules for slice identifiers used in
diffusion/custom and nnU-Net evaluation paths.
"""

import re
from typing import Tuple


_DIFFUSION_SLICE_RE = re.compile(r"^(?P<volume_id>.+)_slice(?P<slice_index>\d+)$")
_NNUNET_SLICE_RE = re.compile(
    r"^(?P<volume_id>.+)_s(?P<slice_index>\d{4})(?:_(?P<modality>\d{4}))?$"
)


def parse_diffusion_slice_identity(raw_identity: str) -> Tuple[str, int]:
    """
    Parse diffusion/custom slice identity.

    Expected normalized basename format:
        "<volume_id>_slice<idx>"

    Args:
        raw_identity: Source path or virtual sample path.

    Returns:
        Tuple of (volume_id, slice_index).
    """
    normalized = _normalize_identity(raw_identity)
    match = _DIFFUSION_SLICE_RE.match(normalized)
    if match is None:
        raise ValueError(
            "Invalid diffusion slice identity "
            f"'{raw_identity}'. Expected '<volume_id>_slice<idx>'."
        )
    volume_id = match.group("volume_id")
    slice_index = int(match.group("slice_index"))
    return volume_id, slice_index


def parse_nnunet_slice_identity(raw_identity: str) -> Tuple[str, int]:
    """
    Parse nnU-Net slice identity.

    Expected normalized basename format:
        "<volume_id>_sXXXX"
    Optionally accepts image-channel filenames:
        "<volume_id>_sXXXX_YYYY"

    Args:
        raw_identity: Source filename, stem, or path.

    Returns:
        Tuple of (volume_id, slice_index).
    """
    normalized = _normalize_identity(raw_identity)
    match = _NNUNET_SLICE_RE.match(normalized)
    if match is None:
        raise ValueError(
            "Invalid nnU-Net slice identity "
            f"'{raw_identity}'. Expected '<volume_id>_sXXXX'."
        )
    volume_id = match.group("volume_id")
    slice_index = int(match.group("slice_index"))
    return volume_id, slice_index


def _normalize_identity(raw_identity: str) -> str:
    """
    Normalize an identity string to a basename without NIfTI extension.
    """
    if raw_identity is None:
        raise ValueError("Identity must not be None.")
    candidate = str(raw_identity).strip()
    if not candidate:
        raise ValueError("Identity must not be empty.")
    normalized_path = candidate.replace("\\", "/")
    name = normalized_path.rsplit("/", 1)[-1]
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return name
