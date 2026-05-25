"""
ISLES24 dataset loaders and parsing helpers.

This module owns the ISLES24-specific data primitives used by the
compatibility facade at `src.data.loaders`.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import threading
from typing import Any, Mapping, Optional

import nibabel
import numpy as np
import torch
import tqdm
from monai.transforms import EnsureTyped, RandCropByPosNegLabeld, Resize, SpatialPadd

from src.data.augmentation import AugmentationPipeline2D, AugmentationPipeline3D
from src.data.modalities import get_modality_params
from src.data.modalities import process_cbf
from src.data.modalities import process_cbv
from src.data.modalities import process_cta
from src.data.modalities import process_mtt
from src.data.modalities import process_ncct
from src.data.modalities import process_tmax
from src.utils.loader_monai_utils import build_monai_compose_safe
from src.utils.loader_utils import LoaderDataUtils

logging.getLogger("nibabel").setLevel(logging.WARNING)

_NNUNET_SLICE_STEM_RE = re.compile(r"^(?P<volume_id>.+)_s(?P<slice_index>\d{4})$")

# A dictionary to map modality names to their processing functions
MODALITY_PROCESSORS = {
    "NCCT": process_ncct,
    "CTA": process_cta,
    "CBF": process_cbf,
    "CBV": process_cbv,
    "MTT": process_mtt,
    "TMAX": process_tmax,
}


def datafold_read(datalist, basedir, fold=0, key="training"):
    """
    Reads and parses the JSON datalist file to create file paths for training and validation sets.
    """
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]
    for d in json_data:
        for k, v in d.items():
            if k == "caseID":
                d[k] = LoaderDataUtils.normalize_case_id(v, field_name="caseID")
            elif isinstance(v, (str, list)):
                d[k] = LoaderDataUtils.resolve_path_value(basedir, v)

    tr = [d for d in json_data if d.get("fold") != fold]
    val = [d for d in json_data if d.get("fold") == fold]

    return tr, val


def _normalize_nnunet_slice_array(array: np.ndarray, source_path: str) -> np.ndarray:
    """
    Normalize nnUNet 2D slice arrays to [H, W].

    Supported source shapes:
    - [H, W]
    - [H, W, 1]
    - [1, H, W]
    """
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[-1] == 1:
        return array[..., 0]
    if array.ndim == 3 and array.shape[0] == 1:
        return array[0, ...]
    raise ValueError(
        f"Unsupported nnUNet slice shape {tuple(array.shape)} at {source_path}. "
        "Expected [H,W], [H,W,1], or [1,H,W]."
    )


class ISLES24Dataset3D(torch.utils.data.Dataset):
    """
    Dataset class for ISLES24 that returns entire 3D volumes.
    Inspired by BRATSDataset.
    """

    def __init__(
        self,
        directory,
        datalist_json,
        fold=0,
        transform=None,
        modalities=None,
        test_flag=False,
        image_size=32,
        preprocessing_configs: Optional[Mapping[str, Any]] = None,
        aug_cfg=None,
        is_training=False,
    ):
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.transform = transform
        self.modalities = modalities if modalities is not None else []
        self.base_modalities = sorted(
            list(
                set(
                    LoaderDataUtils.get_base_modality_key(modality)
                    for modality in self.modalities
                )
            )
        )
        self.image_size = int(image_size)
        if not isinstance(preprocessing_configs, Mapping):
            raise ValueError(
                "ISLES24Dataset3D requires dataset.preprocessing_configs mapping. "
                f"Got: {type(preprocessing_configs).__name__}."
            )
        self.preprocessing_configs = dict(preprocessing_configs)
        self.aug_cfg = aug_cfg
        self.is_training = bool(is_training)
        self.augmentation = None
        if self.is_training and self.aug_cfg is not None:
            self.augmentation = AugmentationPipeline3D(self.aug_cfg)
            print("Initialized 3D augmentation pipeline for ISLES24 training dataset")

        train_files, val_files = datafold_read(datalist=datalist_json, basedir=self.directory, fold=fold)
        self.database = val_files if test_flag else train_files

    def __len__(self):
        return len(self.database)

    def _process_modalities(self, data):
        """Process each modality based on its configuration."""
        processed_images = {}
        for modality_config in self.modalities:
            base_modality = LoaderDataUtils.get_base_modality_key(modality_config)
            raw_data = data[base_modality]

            # RAW mode: not implemented for 3D - use processors.py pipelines instead
            if modality_config.endswith("_RAW"):
                raise NotImplementedError(
                    f"RAW mode '{modality_config}' is not supported for ISLES24Dataset3D. "
                    "Use src/data/processors.py pipelines for 3D volume processing."
                )

            data_stats = LoaderDataUtils.compute_tensor_data_stats(raw_data)

            _base_modality, params = get_modality_params(modality_config, data_stats)

            processor = MODALITY_PROCESSORS.get(base_modality)
            if not processor:
                raise ValueError(f"Unknown base modality: {base_modality}")

            processed = processor(raw_data, **params)
            processed_images[f"processed_{modality_config}"] = processed
        return processed_images

    def __getitem__(self, x):
        filedict = self.database[x]

        # Load all required base modalities and the label
        data = {}
        case_input_paths = LoaderDataUtils.resolve_case_input_paths(
            filedict=filedict,
            base_modalities=self.base_modalities,
        )
        for key, filepath in case_input_paths.items():
            if os.path.exists(filepath):
                nib_img = nibabel.load(filepath)
                data[key] = torch.tensor(nib_img.get_fdata(), dtype=torch.float32)

        # Process modalities to get normalized channels
        processed_images = self._process_modalities(data)

        # Stack processed channels to form the final image
        image_channels = [processed_images[f"processed_{m}"] for m in self.modalities]
        image = torch.stack(image_channels, dim=0)

        label = data.get("label")
        if label is None:  # Handle test sets with no labels
            label = torch.zeros_like(image_channels[0]).unsqueeze(0)
        else:
            label = label.unsqueeze(0)

        if self.augmentation is not None:
            data_dict = {"image": image, "label": label}
            data_dict = self.augmentation(data_dict)
            image = data_dict["image"]
            label = data_dict["label"]

        if self.transform:
            state = torch.get_rng_state()
            image = self.transform(image)
            torch.set_rng_state(state)
            label = self.transform(label)

        return image, label, filedict["caseID"]


def _build_isles24_random_patch_sampler(preprocessing_configs: Mapping[str, Any], patches_per_volume: int):
    random_patch_cfg = preprocessing_configs["random_patches_3d"]
    roi_3d = LoaderDataUtils.as_int_tuple(
        preprocessing_configs["roi"]["volume_3d"],
        expected_len=3,
        field_name="dataset.preprocessing_configs.roi.volume_3d",
    )
    if patches_per_volume <= 0:
        raise ValueError(
            "dataset.preprocessing_configs.random_patches_3d.patches_per_volume.train "
            f"must be > 0, got {patches_per_volume}."
        )

    crop_cfg = random_patch_cfg["rand_crop_by_pos_neg_label"]
    transforms = [
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=roi_3d,
            pos=float(crop_cfg["pos"]),
            neg=float(crop_cfg["neg"]),
            num_samples=patches_per_volume,
            image_key="image",
            image_threshold=float(crop_cfg["image_threshold"]),
            allow_smaller=bool(crop_cfg["allow_smaller"]),
        )
    ]
    if bool(random_patch_cfg["pad_to_divisible"]):
        transforms.append(SpatialPadd(keys=["image", "label"], spatial_size=roi_3d))
    transforms.append(EnsureTyped(keys=["image", "label"], dtype=torch.float32))
    return build_monai_compose_safe(transforms)


class ISLES24RandomPatches3D(torch.utils.data.Dataset):
    """
    ISLES24 random 3D patch dataset.

    Flatten policy:
    - __len__ = num_cases * patches_per_volume.train
    - __getitem__ returns one patch sample
    """

    def __init__(
        self,
        directory,
        datalist_json,
        fold=0,
        transform=None,
        modalities=None,
        test_flag=False,
        image_size=32,
        preprocessing_configs: Optional[Mapping[str, Any]] = None,
        aug_cfg=None,
        is_training=False,
    ):
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.transform = transform
        self.modalities = modalities if modalities is not None else []
        self.base_modalities = sorted(
            list(
                set(
                    LoaderDataUtils.get_base_modality_key(modality)
                    for modality in self.modalities
                )
            )
        )
        self.image_size = int(image_size)
        self.test_flag = bool(test_flag)
        self.aug_cfg = aug_cfg
        self.is_training = bool(is_training)
        if not isinstance(preprocessing_configs, Mapping):
            raise ValueError(
                "ISLES24RandomPatches3D requires dataset.preprocessing_configs mapping. "
                f"Got: {type(preprocessing_configs).__name__}."
            )
        self.preprocessing_configs = dict(preprocessing_configs)

        train_files, val_files = datafold_read(
            datalist=datalist_json,
            basedir=self.directory,
            fold=fold,
        )
        self.database = val_files if self.test_flag else train_files
        self.patches_per_volume = int(
            self.preprocessing_configs["random_patches_3d"]["patches_per_volume"]["train"]
        )
        self.patch_sampler_pipeline = _build_isles24_random_patch_sampler(
            self.preprocessing_configs,
            patches_per_volume=self.patches_per_volume,
        )

        self.augmentation = None
        if self.is_training and self.aug_cfg is not None:
            self.augmentation = AugmentationPipeline3D(self.aug_cfg)
            print("Initialized 3D augmentation pipeline for ISLES24 random-patch training dataset")

    def __len__(self):
        return len(self.database) * self.patches_per_volume

    def _process_modalities(self, data):
        processed_images = {}
        for modality_config in self.modalities:
            base_modality = LoaderDataUtils.get_base_modality_key(modality_config)
            raw_data = data[base_modality]

            if modality_config.endswith("_RAW"):
                raise NotImplementedError(
                    f"RAW mode '{modality_config}' is not supported for ISLES24RandomPatches3D. "
                    "Use src/data/processors.py pipelines for 3D volume processing."
                )

            data_stats = LoaderDataUtils.compute_tensor_data_stats(raw_data)
            _base_modality, params = get_modality_params(modality_config, data_stats)

            processor = MODALITY_PROCESSORS.get(base_modality)
            if not processor:
                raise ValueError(f"Unknown base modality: {base_modality}")
            processed_images[f"processed_{modality_config}"] = processor(raw_data, **params)
        return processed_images

    def _load_case_volume(self, filedict: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        data = {}
        case_input_paths = LoaderDataUtils.resolve_case_input_paths(
            filedict=filedict,
            base_modalities=self.base_modalities,
        )
        for key, filepath in case_input_paths.items():
            if os.path.exists(filepath):
                nib_img = nibabel.load(filepath)
                data[key] = torch.tensor(nib_img.get_fdata(), dtype=torch.float32)

        processed_images = self._process_modalities(data)
        image_channels = [processed_images[f"processed_{m}"] for m in self.modalities]
        image = torch.stack(image_channels, dim=0)

        label = data.get("label")
        if label is None:
            label = torch.zeros_like(image_channels[0]).unsqueeze(0)
        else:
            label = label.unsqueeze(0)
        return {"image": image, "label": label}

    def _load_case_patches(self, filedict: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        data_dict = self._load_case_volume(filedict)
        if self.augmentation is not None:
            data_dict = self.augmentation(data_dict)
        patch_samples = LoaderDataUtils.as_sample_list(self.patch_sampler_pipeline(data_dict))
        if len(patch_samples) != self.patches_per_volume:
            raise RuntimeError(
                "Random patch pipeline returned unexpected number of samples. "
                f"Expected {self.patches_per_volume}, got {len(patch_samples)}."
            )
        return patch_samples

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Patch index out of range: {index}. Dataset length is {len(self)}."
            )
        case_idx = int(index) // self.patches_per_volume
        patch_idx = int(index) % self.patches_per_volume
        filedict = self.database[case_idx]

        patch_sample = self._load_case_patches(filedict)[patch_idx]
        image = patch_sample["image"]
        label = patch_sample["label"]

        if self.transform:
            state = torch.get_rng_state()
            image = self.transform(image)
            torch.set_rng_state(state)
            label = self.transform(label)

        patch_path = f"{filedict['caseID']}_patch{int(patch_idx)}"
        return image, label, patch_path


class ISLES24Dataset2D(torch.utils.data.Dataset):
    """
    Dataset class for ISLES24 that returns 2D slices from 3D volumes.
    Inspired by BRATSDataset3D and CustomDataset3D.
    """

    def __init__(
        self,
        directory,
        datalist_json,
        fold=0,
        transform=None,
        modalities=None,
        test_flag=False,
        image_size=32,
        use_caching=False,
        shared_cache=None,
        cache_lock=None,
        aug_cfg=None,
        is_training=False,
        preprocessing_configs: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.transform = transform
        self.modalities = modalities if modalities is not None else []
        self.base_modalities = sorted(
            list(
                set(
                    LoaderDataUtils.get_base_modality_key(modality)
                    for modality in self.modalities
                )
            )
        )

        train_files, val_files = datafold_read(datalist=datalist_json, basedir=self.directory, fold=fold)
        self.database = val_files if test_flag else train_files

        self.all_slices = []

        self.image_size = int(image_size)
        if not isinstance(preprocessing_configs, Mapping):
            raise ValueError(
                "ISLES24Dataset2D requires dataset.preprocessing_configs mapping. "
                f"Got: {type(preprocessing_configs).__name__}."
            )
        self.preprocessing_configs = dict(preprocessing_configs)
        self.use_caching = use_caching
        self._cache_prefix = "ts" if test_flag else "tr"

        # NEW: Initialize augmentation pipeline
        self.aug_cfg = aug_cfg
        self.is_training = is_training
        self.augmentation = None

        if self.is_training and self.aug_cfg is not None:
            self.augmentation = AugmentationPipeline2D(self.aug_cfg)
            print("Initialized augmentation pipeline for training dataset")

        print("Pre-calculating dataset size...")
        for case_idx, filedict in tqdm.tqdm(enumerate(self.database), total=len(self.database)):
            # Use the first modality to determine the number of slices
            first_mod_key = self.base_modalities[0]
            filepath = filedict[first_mod_key][0] if isinstance(filedict[first_mod_key], list) else filedict[first_mod_key]
            if os.path.exists(filepath):
                num_slices = nibabel.load(filepath).shape[-1]
                self.all_slices.extend([(case_idx, slice_idx) for slice_idx in range(num_slices)])

        self.cache = None
        self.cache_lock = None
        if self.use_caching:
            if shared_cache is not None:
                self.cache = shared_cache
                self.cache_lock = cache_lock or threading.Lock()
            else:
                self.cache = {}
                self.cache_lock = threading.Lock()

    def __len__(self):
        return len(self.all_slices)

    def _process_modalities(self, data_slice):
        """Process each 2D modality slice based on its configuration."""
        processed_images = {}
        for modality_config in self.modalities:
            base_modality = LoaderDataUtils.get_base_modality_key(modality_config)
            raw_data = data_slice[base_modality]

            # RAW mode: skip normalization, passthrough raw intensity values
            # Used for nnU-Net export where nnU-Net handles its own normalization
            if modality_config.endswith("_RAW"):
                processed_images[f"processed_{modality_config}"] = raw_data.float()
                continue

            data_stats = LoaderDataUtils.compute_tensor_data_stats(raw_data)

            _base_modality, params = get_modality_params(modality_config, data_stats)

            processor = MODALITY_PROCESSORS.get(base_modality)
            if not processor:
                raise ValueError(f"Unknown base modality: {base_modality}")

            # Assuming processor can handle 2D Tensors
            processed = processor(raw_data, **params)
            processed_images[f"processed_{modality_config}"] = processed
        return processed_images

    def __getitem__(self, x):
        case_idx, slice_idx = self.all_slices[x]
        filedict = self.database[case_idx]
        cache_key = (self._cache_prefix, int(case_idx))

        if self.use_caching:
            if cache_key not in self.cache:
                with self.cache_lock:
                    if cache_key not in self.cache:  # Double-check after acquiring lock
                        data = {}
                        case_input_paths = LoaderDataUtils.resolve_case_input_paths(
                            filedict=filedict,
                            base_modalities=self.base_modalities,
                        )
                        for key, filepath in case_input_paths.items():
                            if os.path.exists(filepath):
                                nib_img = nibabel.load(filepath)
                                data[key] = torch.from_numpy(
                                    nib_img.get_fdata().astype(np.float32)
                                )
                        self.cache[cache_key] = data

            data_slice = {}
            for key in self.cache[cache_key]:
                data_slice[key] = self.cache[cache_key][key][..., slice_idx]
        else:
            # Load directly without caching
            data_slice = {}
            case_input_paths = LoaderDataUtils.resolve_case_input_paths(
                filedict=filedict,
                base_modalities=self.base_modalities,
            )
            for key, filepath in case_input_paths.items():
                if os.path.exists(filepath):
                    nib_img = nibabel.load(filepath)
                    vol_data = torch.from_numpy(nib_img.get_fdata().astype(np.float32))
                    data_slice[key] = vol_data[..., slice_idx]

        processed_images = self._process_modalities(data_slice)

        image_channels = [processed_images[f"processed_{m}"] for m in self.modalities]
        image = torch.stack(image_channels, dim=0)

        label = data_slice.get("label")
        if label is None:
            label = torch.zeros_like(image_channels[0]).unsqueeze(0)
        else:
            label = label.unsqueeze(0)

        # Resize to match model input size
        resizer = Resize(spatial_size=(self.image_size, self.image_size))
        image = resizer(image)
        label = resizer(label)

        # NEW: Apply augmentation if training mode
        if self.augmentation is not None:
            data_dict = {"image": image, "label": label}
            data_dict = self.augmentation(data_dict)
            image = data_dict["image"]
            label = data_dict["label"]

        if self.transform:
            state = torch.get_rng_state()
            image = self.transform(image)
            torch.set_rng_state(state)
            label = self.transform(label)

        virtual_path = f"{filedict['caseID']}_slice{slice_idx}"
        return image, label, virtual_path


class ISLES24NNUNet2D(torch.utils.data.Dataset):
    """
    Dataset class for ISLES24 nnUNet-precomputed 2D slices.

    Expects nnUNet-style structure:
      Dataset{dataset_id}_{dataset_name}/
        imagesTr/*.nii.gz
        labelsTr/*.nii.gz
        imagesTs/*.nii.gz
        labelsTs/*.nii.gz

    Image channel files are expected as:
      <volume_id>_sXXXX_0000.nii.gz, <volume_id>_sXXXX_0001.nii.gz, ...
    Label files are expected as:
      <volume_id>_sXXXX.nii.gz
    """

    def __init__(
        self,
        nnunet_root,
        dataset_id,
        dataset_name,
        modalities=None,
        test_flag=False,
        image_size=32,
        transform=None,
        use_caching=False,
        shared_cache=None,
        cache_lock=None,
        aug_cfg=None,
        is_training=False,
        per_side_context_slices=0,
        channel_layout="slice_major",
    ):
        super().__init__()
        self.nnunet_root = os.path.expanduser(str(nnunet_root))
        self.dataset_id = str(dataset_id)
        self.dataset_name = str(dataset_name)
        self.modalities = modalities if modalities is not None else []
        self.num_modalities = len(self.modalities)
        self.test_flag = bool(test_flag)
        self.image_size = int(image_size)
        self.transform = transform
        self.use_caching = bool(use_caching)
        self.return_metadata = False
        self._cache_prefix = "ts" if self.test_flag else "tr"
        self.per_side_context_slices = int(per_side_context_slices)
        self.channel_layout = str(channel_layout)
        self.num_effective_slices = (2 * self.per_side_context_slices) + 1
        self.effective_input_channels = self.num_modalities * self.num_effective_slices

        if self.num_modalities <= 0:
            raise ValueError("ISLES24NNUNet2D requires a non-empty modalities list.")
        if self.per_side_context_slices < 0:
            raise ValueError("per_side_context_slices must be >= 0 for ISLES24NNUNet2D.")
        if self.channel_layout not in {"slice_major", "modality_major"}:
            raise ValueError(
                "channel_layout must be one of {'slice_major', 'modality_major'} "
                f"for ISLES24NNUNet2D, got: {self.channel_layout}"
            )

        dataset_dir_name = f"Dataset{self.dataset_id}_{self.dataset_name}"
        self.dataset_dir = os.path.join(self.nnunet_root, dataset_dir_name)
        split_suffix = "Ts" if self.test_flag else "Tr"
        self.images_dir = os.path.join(self.dataset_dir, f"images{split_suffix}")
        self.labels_dir = os.path.join(self.dataset_dir, f"labels{split_suffix}")

        if not os.path.isdir(self.dataset_dir):
            raise FileNotFoundError(
                f"nnUNet dataset folder not found: {self.dataset_dir}. "
                f"Expected root='{self.nnunet_root}', dataset='{dataset_dir_name}'."
            )
        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(f"Missing nnUNet image directory: {self.images_dir}")
        if not os.path.isdir(self.labels_dir):
            raise FileNotFoundError(f"Missing nnUNet label directory: {self.labels_dir}")

        label_paths = sorted(glob.glob(os.path.join(self.labels_dir, "*.nii.gz")))
        if len(label_paths) == 0:
            raise FileNotFoundError(
                f"No label slices found in {self.labels_dir}. "
                "Cannot build nnUNet 2D dataset split."
            )

        self.samples = []
        for label_path in label_paths:
            label_name = os.path.basename(label_path)
            slice_stem = label_name.replace(".nii.gz", "")
            match = _NNUNET_SLICE_STEM_RE.match(slice_stem)
            if match is None:
                raise ValueError(
                    f"Invalid nnUNet slice label name '{label_name}'. "
                    "Expected '<volume_id>_sXXXX.nii.gz'."
                )

            volume_id = match.group("volume_id")
            slice_index = int(match.group("slice_index"))
            image_paths = [
                os.path.join(self.images_dir, f"{slice_stem}_{channel_idx:04d}.nii.gz")
                for channel_idx in range(self.num_modalities)
            ]
            missing = [path for path in image_paths if not os.path.exists(path)]
            if missing:
                raise FileNotFoundError(
                    "Missing nnUNet image channels for sample "
                    f"'{slice_stem}'. Expected {self.num_modalities} channels, "
                    f"missing {len(missing)} file(s): {missing[:3]}"
                    f"{' ...' if len(missing) > 3 else ''}"
                )

            self.samples.append(
                {
                    "slice_stem": slice_stem,
                    "volume_id": volume_id,
                    "slice_index": slice_index,
                    "label_path": label_path,
                    "image_paths": image_paths,
                    "virtual_path": f"{volume_id}_slice{slice_index}",
                }
            )

        # Fast lookup for neighboring slices within the same volume.
        self._sample_lookup = {}
        self._volume_slice_bounds = {}
        for sample in self.samples:
            volume_id = sample["volume_id"]
            slice_index = int(sample["slice_index"])
            key = (volume_id, slice_index)
            self._sample_lookup[key] = sample
            if volume_id not in self._volume_slice_bounds:
                self._volume_slice_bounds[volume_id] = [slice_index, slice_index]
            else:
                self._volume_slice_bounds[volume_id][0] = min(self._volume_slice_bounds[volume_id][0], slice_index)
                self._volume_slice_bounds[volume_id][1] = max(self._volume_slice_bounds[volume_id][1], slice_index)

        self.aug_cfg = aug_cfg
        self.is_training = bool(is_training)
        self.augmentation = None
        if self.is_training and self.aug_cfg is not None:
            self.augmentation = AugmentationPipeline2D(self.aug_cfg)
            print("Initialized augmentation pipeline for nnUNet training dataset")

        self.cache = None
        self.cache_lock = None
        if self.use_caching:
            if shared_cache is not None:
                self.cache = shared_cache
                self.cache_lock = cache_lock or threading.Lock()
            else:
                self.cache = {}
                self.cache_lock = threading.Lock()

    def __len__(self):
        return len(self.samples)

    def _resolve_neighbor_sample(self, sample, relative_offset):
        volume_id = sample["volume_id"]
        center_idx = int(sample["slice_index"])
        target_idx = center_idx + int(relative_offset)
        min_idx, max_idx = self._volume_slice_bounds[volume_id]
        if target_idx < min_idx or target_idx > max_idx:
            return None
        neighbor = self._sample_lookup.get((volume_id, target_idx))
        if neighbor is None:
            raise ValueError(
                f"Missing neighbor sample for volume '{volume_id}' at slice index "
                f"{target_idx} while building context for '{sample['slice_stem']}'."
            )
        return neighbor

    def _load_modality_stack(self, sample, expected_shape_hw=None):
        modality_tensors = []
        sample_shape = tuple(expected_shape_hw) if expected_shape_hw is not None else None

        for image_path in sample["image_paths"]:
            image_np = nibabel.load(image_path).get_fdata().astype(np.float32)
            image_np = _normalize_nnunet_slice_array(image_np, image_path)
            if sample_shape is None:
                sample_shape = tuple(image_np.shape)
            elif tuple(image_np.shape) != sample_shape:
                raise ValueError(
                    f"Channel geometry mismatch for '{sample['slice_stem']}'. "
                    f"Expected {sample_shape}, got {tuple(image_np.shape)} at {image_path}."
                )
            modality_tensors.append(torch.from_numpy(image_np))

        if sample_shape is None:
            raise RuntimeError(f"No image channels loaded for sample '{sample['slice_stem']}'.")
        image = torch.stack(modality_tensors, dim=0).float()
        return image, sample_shape

    def _flatten_context_channels(self, context_modalities):
        if self.channel_layout == "slice_major":
            return torch.cat(context_modalities, dim=0)

        stacked = torch.stack(context_modalities, dim=0)  # [S, M, H, W]
        stacked = stacked.permute(1, 0, 2, 3).contiguous()  # [M, S, H, W]
        return stacked.view(
            self.num_modalities * self.num_effective_slices,
            stacked.shape[-2],
            stacked.shape[-1],
        )

    def _load_raw_sample(self, sample):
        label_nib = nibabel.load(sample["label_path"])
        label_np = label_nib.get_fdata().astype(np.float32)
        label_np = _normalize_nnunet_slice_array(label_np, sample["label_path"])
        sample_shape = tuple(label_np.shape)

        center_modalities, center_shape = self._load_modality_stack(
            sample, expected_shape_hw=sample_shape
        )
        if tuple(center_shape) != sample_shape:
            raise ValueError(
                f"Image/label geometry mismatch for '{sample['slice_stem']}'. "
                f"image={tuple(center_shape)}, label={sample_shape}."
            )

        context_modalities = []
        for relative_offset in range(-self.per_side_context_slices, self.per_side_context_slices + 1):
            if relative_offset == 0:
                context_modalities.append(center_modalities)
                continue

            neighbor_sample = self._resolve_neighbor_sample(sample, relative_offset)
            if neighbor_sample is None:
                zero_modalities = torch.zeros(
                    (self.num_modalities, sample_shape[0], sample_shape[1]),
                    dtype=torch.float32,
                )
                context_modalities.append(zero_modalities)
                continue

            neighbor_modalities, _ = self._load_modality_stack(
                neighbor_sample, expected_shape_hw=sample_shape
            )
            context_modalities.append(neighbor_modalities)

        image = self._flatten_context_channels(context_modalities)
        label = torch.from_numpy(label_np).float().unsqueeze(0)

        zooms = label_nib.header.get_zooms()
        spacing_xyz = [
            float(zooms[0]) if len(zooms) > 0 else 1.0,
            float(zooms[1]) if len(zooms) > 1 else 1.0,
            float(zooms[2]) if len(zooms) > 2 else 1.0,
        ]
        axcodes = [str(code) for code in nibabel.aff2axcodes(label_nib.affine)]
        pre_h, pre_w = int(label.shape[-2]), int(label.shape[-1])
        slice_meta = {
            "source_path": sample["label_path"],
            "source_affine": label_nib.affine.tolist(),
            "source_spacing_xyz": spacing_xyz,
            "source_axcodes": axcodes,
            "source_volume_shape": [pre_h, pre_w, 1],
            "slice_axis": 2,
            "per_side_context_slices": int(self.per_side_context_slices),
            "num_effective_slices": int(self.num_effective_slices),
            "effective_input_channels": int(self.effective_input_channels),
            "channel_layout": self.channel_layout,
            "pre_resize_shape_hw": [pre_h, pre_w],
            "post_resize_shape_hw": [self.image_size, self.image_size],
            "source_identity": sample["slice_stem"],
        }
        return image, label, sample["virtual_path"], slice_meta

    def __getitem__(self, x):
        sample = self.samples[x]
        cache_key = (self._cache_prefix, int(x))

        if self.use_caching:
            if cache_key not in self.cache:
                with self.cache_lock:
                    if cache_key not in self.cache:
                        self.cache[cache_key] = self._load_raw_sample(sample)
            image, label, virtual_path, slice_meta = self.cache[cache_key]
            image = image.clone()
            label = label.clone()
            slice_meta = dict(slice_meta)
        else:
            image, label, virtual_path, slice_meta = self._load_raw_sample(sample)

        # Keep image-size contract consistent with online loader.
        resizer = Resize(spatial_size=(self.image_size, self.image_size))
        image = resizer(image)
        label = resizer(label)

        if self.augmentation is not None:
            data_dict = {"image": image, "label": label}
            data_dict = self.augmentation(data_dict)
            image = data_dict["image"]
            label = data_dict["label"]

        if self.transform:
            state = torch.get_rng_state()
            image = self.transform(image)
            torch.set_rng_state(state)
            label = self.transform(label)

        if self.return_metadata:
            return image, label, virtual_path, slice_meta
        return image, label, virtual_path


# Explicit naming for loader routing contracts.
ISLES24OnlineProc2D = ISLES24Dataset2D

__all__ = [
    "datafold_read",
    "ISLES24Dataset3D",
    "ISLES24RandomPatches3D",
    "ISLES24Dataset2D",
    "ISLES24NNUNet2D",
    "ISLES24OnlineProc2D",
]
