"""
Shared MONAI composition helpers used by loader stacks.
"""

from __future__ import annotations

from typing import Any, Sequence

import monai.transforms.compose as monai_compose_module
import monai.transforms.transform as monai_transform_module
import numpy as np
from monai.transforms import Compose


def build_monai_compose_safe(transforms: Sequence[Any]) -> Compose:
    """
    Build MONAI Compose with a compatibility guard for uint32 seed overflow.

    Some NumPy/MONAI combos raise on `_seed % MAX_SEED` when MAX_SEED is set to
    2**32. Capping module MAX_SEED values to uint32 max preserves native MONAI
    behavior while avoiding overflow.
    """
    uint32_max = int(np.iinfo(np.uint32).max)
    if getattr(monai_compose_module, "MAX_SEED", uint32_max) > uint32_max:
        monai_compose_module.MAX_SEED = uint32_max
    if getattr(monai_transform_module, "MAX_SEED", uint32_max) > uint32_max:
        monai_transform_module.MAX_SEED = uint32_max
    return Compose(list(transforms))
