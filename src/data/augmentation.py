"""
Data augmentation pipeline for 2D medical imaging.

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

from typing import Dict, List, Optional
from omegaconf import DictConfig
import torch
import numpy as np
import random
from monai import transforms as mt


class AugmentationPipeline2D:
    """
    Builds and applies MONAI-based 2D augmentation pipeline from Hydra config.
    
    This class encapsulates all augmentation logic for the MedSegDiff training pipeline.
    It processes data dicts with 'image' and 'label' keys, applying spatial and intensity
    transforms based on the provided configuration.
    
    Args:
        aug_cfg: Augmentation configuration from configs/augmentation/*.yaml
    
    Attributes:
        aug_cfg: Stored configuration
        transform: Composed MONAI transform pipeline (None if no augmentations enabled)
    
    Example:
        >>> cfg = OmegaConf.load('configs/augmentation/light_2d.yaml')
        >>> pipeline = AugmentationPipeline2D(cfg)
        >>> 
        >>> # Apply to single sample
        >>> data = {'image': torch.randn(2, 128, 128), 'label': torch.randint(0, 2, (1, 128, 128))}
        >>> augmented = pipeline(data)
        >>> 
        >>> # Pipeline is callable and can be used in dataset __getitem__
        >>> if self.augmentation is not None:
        >>>     data_dict = self.augmentation(data_dict)
    """
    
    def __init__(self, aug_cfg: DictConfig):
        """Initialize augmentation pipeline from config."""
        self.aug_cfg = aug_cfg
        self.transform = self._build_transform()
    
    def _build_transform(self) -> Optional[mt.Compose]:
        """
        Build unified MONAI Compose pipeline from config.
        
        Returns:
            Composed transform pipeline, or None if all augmentations disabled
        """
        transforms_list = []
        
        # Add spatial augmentations
        spatial_transforms = self._build_spatial_transforms()
        transforms_list.extend(spatial_transforms)
        
        # Add intensity augmentations
        intensity_transforms = self._build_intensity_transforms()
        transforms_list.extend(intensity_transforms)
        
        # Return composed pipeline, or None if no transforms
        if len(transforms_list) == 0:
            return None
        
        try:
            return mt.Compose(transforms_list, map_items=False)
        except OverflowError:
            # Workaround for MONAI overflow issue on some systems/versions
            # Fallback to simple sequential execution
            print("Warning: MONAI Compose OverflowError caught. Using fallback composition.")
            
            class SimpleCompose:
                def __init__(self, transforms):
                    self.transforms = transforms
                
                def set_random_state(self, seed=None, state=None):
                    for t in self.transforms:
                        if hasattr(t, 'set_random_state'):
                            t.set_random_state(seed=seed, state=state)

                def __call__(self, data):
                    for t in self.transforms:
                        data = t(data)
                    return data
            
            return SimpleCompose(transforms_list)
    
    def _build_spatial_transforms(self) -> List:
        """
        Build spatial transforms that apply to both image and label.
        
        Spatial transforms (flip, rotate) must be synchronized between image and label
        to maintain anatomical consistency. MONAI's dictionary-based transforms with
        shared keys handle this automatically.
        
        Returns:
            List of MONAI transform instances for spatial augmentations
        """
        if not self.aug_cfg.spatial.enabled:
            return []
        
        transforms_list = []
        spatial_cfg = self.aug_cfg.spatial
        
        # Random flip (required when spatial enabled)
        if spatial_cfg.random_flip.enabled:
            transforms_list.append(
                mt.RandFlipd(
                    keys=['image', 'label'],
                    prob=spatial_cfg.random_flip.prob,
                    spatial_axis=spatial_cfg.random_flip.spatial_axis
                )
            )
        
        # Random rotation (optional)
        if 'random_rotate' in spatial_cfg and spatial_cfg.random_rotate.enabled:
            rot_cfg = spatial_cfg.random_rotate
            transforms_list.append(
                mt.RandRotated(
                    keys=['image', 'label'],
                    prob=rot_cfg.prob,
                    range_x=rot_cfg.range_x,
                    mode=(rot_cfg.mode, 'nearest'),  # bilinear for image, nearest for label
                    padding_mode=rot_cfg.padding_mode
                )
            )
        
        return transforms_list
    
    def _build_intensity_transforms(self) -> List:
        """
        Build intensity transforms that apply to image only.
        
        Intensity transforms (shift, scale, noise, blur) modify pixel values but should
        NOT be applied to segmentation labels (which are categorical/binary).
        
        Returns:
            List of MONAI transform instances for intensity augmentations
        """
        if not self.aug_cfg.intensity.enabled:
            return []
        
        transforms_list = []
        intensity_cfg = self.aug_cfg.intensity
        
        # Random intensity shift (required when intensity enabled)
        if intensity_cfg.random_shift.enabled:
            transforms_list.append(
                mt.RandShiftIntensityd(
                    keys=['image'],  # Only apply to image, not label
                    offsets=intensity_cfg.random_shift.offsets,
                    prob=intensity_cfg.random_shift.prob
                )
            )
        
        # Random intensity scale (required when intensity enabled)
        if intensity_cfg.random_scale.enabled:
            transforms_list.append(
                mt.RandScaleIntensityd(
                    keys=['image'],
                    factors=intensity_cfg.random_scale.factors,
                    prob=intensity_cfg.random_scale.prob
                )
            )
        
        # Random gamma (optional)
        if 'random_gamma' in intensity_cfg and intensity_cfg.random_gamma.enabled:
            gamma_cfg = intensity_cfg.random_gamma
            transforms_list.append(
                mt.RandAdjustContrastd(
                    keys=['image'],
                    prob=gamma_cfg.prob,
                    gamma=gamma_cfg.gamma_range
                )
            )
        
        # Random Gaussian noise (optional)
        if 'random_gaussian_noise' in intensity_cfg and intensity_cfg.random_gaussian_noise.enabled:
            noise_cfg = intensity_cfg.random_gaussian_noise
            transforms_list.append(
                mt.RandGaussianNoised(
                    keys=['image'],
                    prob=noise_cfg.prob,
                    mean=noise_cfg.mean,
                    std=noise_cfg.std
                )
            )
        
        # Random Gaussian blur (optional)
        if 'random_gaussian_blur' in intensity_cfg and intensity_cfg.random_gaussian_blur.enabled:
            blur_cfg = intensity_cfg.random_gaussian_blur
            transforms_list.append(
                mt.RandGaussianSmoothd(
                    keys=['image'],
                    prob=blur_cfg.prob,
                    sigma_x=blur_cfg.sigma_range,
                    sigma_y=blur_cfg.sigma_range
                )
            )
        
        return transforms_list
    
    def __call__(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply augmentation pipeline to data dictionary.
        
        Args:
            data_dict: Dictionary with 'image' and 'label' keys
                - image: [C, H, W] tensor (multi-channel, post-modality-processing)
                - label: [1, H, W] tensor (binary segmentation mask)
        
        Returns:
            Augmented data dictionary with same structure
        """
        if self.transform is None:
            return data_dict
        return self.transform(data_dict)

