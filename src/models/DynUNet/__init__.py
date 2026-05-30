"""
DynUNet package for discriminative segmentation.

Provides MONAI DynUNet wrapped for pipeline compatibility.
"""

from .model_adapter import DynUNetAdapter

__all__ = ["DynUNetAdapter"]
