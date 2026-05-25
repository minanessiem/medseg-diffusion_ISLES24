"""
Dimension-aware augmentation pipeline for 2D and 3D medical imaging.

This module provides a configuration-driven augmentation system using MONAI transforms.
All augmentations operate on merged multi-channel images after modality-specific preprocessing.

Design principles:
  - Spatial transforms apply to both image and label (synchronized)
  - Intensity transforms apply to image only (preserve label)
  - All config keys must be explicitly defined (no silent defaults)
  - Missing required keys will raise KeyError naturally

Example usage:
    >>> from omegaconf import OmegaConf
    >>> cfg = OmegaConf.load('configs/augmentation/light_2d.yaml')
    >>> pipeline = AugmentationPipeline2D(cfg)
    >>> data_dict = {'image': image_tensor, 'label': label_tensor}
    >>> augmented = pipeline(data_dict)
"""

from typing import Any, Dict, List, Optional

import monai.transforms.compose as monai_compose_module
import monai.transforms.transform as monai_transform_module
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from monai import transforms as mt


class AugmentationPipeline:
    """
    Build and apply MONAI-based augmentations for either 2D or 3D tensors.

    Args:
        aug_cfg: Augmentation config from `configs/augmentation/*.yaml`.
        spatial_dims: Optional explicit spatial dims (2 or 3). When omitted,
            dims are inferred from the input tensor shape at call time.
    """

    def __init__(self, aug_cfg: DictConfig, spatial_dims: Optional[int] = None):
        self.aug_cfg = aug_cfg
        self.spatial_dims = spatial_dims
        if self.spatial_dims not in (None, 2, 3):
            raise ValueError(
                f"spatial_dims must be one of {{None,2,3}}, got {self.spatial_dims}."
            )

        self._transform_cache: Dict[int, Optional[Any]] = {}
        self.transform = None
        if self.spatial_dims is not None:
            self.transform = self._build_transform(spatial_dims=self.spatial_dims)
            self._transform_cache[self.spatial_dims] = self.transform

    @staticmethod
    def _normalize_spatial_axes(
        configured_axes: Any,
        *,
        spatial_dims: int,
    ) -> List[int]:
        if OmegaConf.is_list(configured_axes) or isinstance(
            configured_axes, (list, tuple, set)
        ):
            axes = [int(axis) for axis in configured_axes]
        else:
            axes = [int(configured_axes)]

        valid_axes = set(range(spatial_dims))
        invalid_axes = [axis for axis in axes if axis not in valid_axes]
        if invalid_axes:
            raise ValueError(
                f"Invalid spatial_axis {invalid_axes} for {spatial_dims}D augmentation. "
                f"Valid axes: {sorted(valid_axes)}."
            )
        return axes

    @staticmethod
    def _build_compose(transforms_list: List[Any]) -> Optional[Any]:
        if len(transforms_list) == 0:
            return None

        # Keep parity with loader-stack MONAI compatibility guard.
        uint32_max = int(np.iinfo(np.uint32).max)
        if getattr(monai_compose_module, "MAX_SEED", uint32_max) > uint32_max:
            monai_compose_module.MAX_SEED = uint32_max
        if getattr(monai_transform_module, "MAX_SEED", uint32_max) > uint32_max:
            monai_transform_module.MAX_SEED = uint32_max

        try:
            return mt.Compose(transforms_list, map_items=False)
        except OverflowError:
            print("Warning: MONAI Compose OverflowError caught. Using fallback composition.")

            class SimpleCompose:
                def __init__(self, transforms):
                    self.transforms = transforms

                def set_random_state(self, seed=None, state=None):
                    for transform in self.transforms:
                        if hasattr(transform, "set_random_state"):
                            transform.set_random_state(seed=seed, state=state)

                def __call__(self, data):
                    for transform in self.transforms:
                        data = transform(data)
                    return data

            return SimpleCompose(transforms_list)

    def _build_spatial_transforms(self, spatial_dims: int) -> List[Any]:
        if not self.aug_cfg.spatial.enabled:
            return []

        transforms_list: List[Any] = []
        spatial_cfg = self.aug_cfg.spatial

        if spatial_cfg.random_flip.enabled:
            flip_axes = self._normalize_spatial_axes(
                spatial_cfg.random_flip.spatial_axis,
                spatial_dims=spatial_dims,
            )
            transforms_list.append(
                mt.RandFlipd(
                    keys=["image", "label"],
                    prob=spatial_cfg.random_flip.prob,
                    spatial_axis=flip_axes,
                )
            )

        if "random_rotate" in spatial_cfg and spatial_cfg.random_rotate.enabled:
            rot_cfg = spatial_cfg.random_rotate
            rotate_kwargs = {
                "keys": ["image", "label"],
                "prob": rot_cfg.prob,
                "mode": (rot_cfg.mode, "nearest"),
                "padding_mode": rot_cfg.padding_mode,
                "range_x": rot_cfg.range_x,
            }
            if spatial_dims == 3:
                rotate_kwargs["range_y"] = (
                    rot_cfg.range_y if "range_y" in rot_cfg else rot_cfg.range_x
                )
                rotate_kwargs["range_z"] = (
                    rot_cfg.range_z if "range_z" in rot_cfg else rot_cfg.range_x
                )
            transforms_list.append(mt.RandRotated(**rotate_kwargs))

        return transforms_list

    def _build_intensity_transforms(self, spatial_dims: int) -> List[Any]:
        if not self.aug_cfg.intensity.enabled:
            return []

        transforms_list: List[Any] = []
        intensity_cfg = self.aug_cfg.intensity

        if intensity_cfg.random_shift.enabled:
            transforms_list.append(
                mt.RandShiftIntensityd(
                    keys=["image"],
                    offsets=intensity_cfg.random_shift.offsets,
                    prob=intensity_cfg.random_shift.prob,
                )
            )

        if intensity_cfg.random_scale.enabled:
            transforms_list.append(
                mt.RandScaleIntensityd(
                    keys=["image"],
                    factors=intensity_cfg.random_scale.factors,
                    prob=intensity_cfg.random_scale.prob,
                )
            )

        if "random_gamma" in intensity_cfg and intensity_cfg.random_gamma.enabled:
            gamma_cfg = intensity_cfg.random_gamma
            transforms_list.append(
                mt.RandAdjustContrastd(
                    keys=["image"],
                    prob=gamma_cfg.prob,
                    gamma=gamma_cfg.gamma_range,
                )
            )

        if (
            "random_gaussian_noise" in intensity_cfg
            and intensity_cfg.random_gaussian_noise.enabled
        ):
            noise_cfg = intensity_cfg.random_gaussian_noise
            transforms_list.append(
                mt.RandGaussianNoised(
                    keys=["image"],
                    prob=noise_cfg.prob,
                    mean=noise_cfg.mean,
                    std=noise_cfg.std,
                )
            )

        if (
            "random_gaussian_blur" in intensity_cfg
            and intensity_cfg.random_gaussian_blur.enabled
        ):
            blur_cfg = intensity_cfg.random_gaussian_blur
            blur_kwargs = {
                "keys": ["image"],
                "prob": blur_cfg.prob,
                "sigma_x": blur_cfg.sigma_range,
                "sigma_y": blur_cfg.sigma_range,
            }
            if spatial_dims == 3:
                blur_kwargs["sigma_z"] = (
                    blur_cfg.sigma_range_z
                    if "sigma_range_z" in blur_cfg
                    else blur_cfg.sigma_range
                )
            transforms_list.append(mt.RandGaussianSmoothd(**blur_kwargs))

        return transforms_list

    def _build_transform(self, *, spatial_dims: int) -> Optional[Any]:
        transforms_list: List[Any] = []
        transforms_list.extend(self._build_spatial_transforms(spatial_dims=spatial_dims))
        transforms_list.extend(
            self._build_intensity_transforms(spatial_dims=spatial_dims)
        )
        return self._build_compose(transforms_list)

    @staticmethod
    def _validate_data_dict(
        data_dict: Dict[str, torch.Tensor],
        *,
        spatial_dims: int,
    ) -> None:
        if "image" not in data_dict or "label" not in data_dict:
            raise KeyError("Augmentation data_dict must contain both 'image' and 'label'.")

        image = data_dict["image"]
        label = data_dict["label"]
        if not isinstance(image, torch.Tensor) or not isinstance(label, torch.Tensor):
            raise TypeError(
                "Augmentation expects torch.Tensor values for 'image' and 'label'."
            )

        expected_rank = spatial_dims + 1
        if image.ndim != expected_rank:
            raise ValueError(
                f"Expected image rank {expected_rank} for {spatial_dims}D augmentation, "
                f"got shape {tuple(image.shape)}."
            )
        if label.ndim != expected_rank:
            raise ValueError(
                f"Expected label rank {expected_rank} for {spatial_dims}D augmentation, "
                f"got shape {tuple(label.shape)}."
            )
        if tuple(image.shape[1:]) != tuple(label.shape[1:]):
            raise ValueError(
                "Image/label spatial shapes must match for augmentation. "
                f"Got image={tuple(image.shape)}, label={tuple(label.shape)}."
            )

    def _resolve_spatial_dims(self, data_dict: Dict[str, torch.Tensor]) -> int:
        if self.spatial_dims is not None:
            return int(self.spatial_dims)

        image = data_dict.get("image")
        if not isinstance(image, torch.Tensor):
            raise TypeError(
                "Augmentation requires tensor input under 'image' when spatial_dims "
                "is not explicitly provided."
            )
        inferred_dims = int(image.ndim) - 1
        if inferred_dims not in (2, 3):
            raise ValueError(
                f"Unable to infer augmentation spatial dims from image shape "
                f"{tuple(image.shape)}. Expected [C,H,W] or [C,H,W,D]."
            )
        return inferred_dims

    def _get_transform(self, spatial_dims: int) -> Optional[Any]:
        if spatial_dims not in self._transform_cache:
            self._transform_cache[spatial_dims] = self._build_transform(
                spatial_dims=spatial_dims
            )
        return self._transform_cache[spatial_dims]

    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        spatial_dims = self._resolve_spatial_dims(data_dict)
        self._validate_data_dict(data_dict, spatial_dims=spatial_dims)
        transform = self._get_transform(spatial_dims)
        if self.spatial_dims is not None:
            self.transform = transform
        if transform is None:
            return data_dict
        return transform(data_dict)


class AugmentationPipeline2D(AugmentationPipeline):
    """Compatibility wrapper that enforces 2D augmentation behavior."""

    def __init__(self, aug_cfg: DictConfig):
        super().__init__(aug_cfg=aug_cfg, spatial_dims=2)


class AugmentationPipeline3D(AugmentationPipeline):
    """Convenience wrapper for explicit 3D augmentation behavior."""

    def __init__(self, aug_cfg: DictConfig):
        super().__init__(aug_cfg=aug_cfg, spatial_dims=3)

