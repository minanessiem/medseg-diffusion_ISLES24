import sys
print("[DEBUG:loaders.py] Starting imports...", flush=True)

import os
import glob
import re
print("[DEBUG:loaders.py] os, glob done", flush=True)

import numpy as np
import pandas as pd
print("[DEBUG:loaders.py] numpy, pandas done", flush=True)

from sklearn.model_selection import train_test_split
print("[DEBUG:loaders.py] sklearn done", flush=True)

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
print("[DEBUG:loaders.py] torch done", flush=True)

import json
import nibabel
print("[DEBUG:loaders.py] json, nibabel done", flush=True)

from monai.transforms import Resize
print("[DEBUG:loaders.py] monai done", flush=True)

from omegaconf import OmegaConf
print("[DEBUG:loaders.py] omegaconf done", flush=True)

from src.data.processors import get_image_transform, get_mask_transform, get_joint_transform
print("[DEBUG:loaders.py] processors done", flush=True)

from src.data.modalities import get_modality_params
from src.data.modalities import process_cbf
from src.data.modalities import process_cbv
from src.data.modalities import process_cta
from src.data.modalities import process_mtt
from src.data.modalities import process_ncct
from src.data.modalities import process_tmax
print("[DEBUG:loaders.py] modalities done", flush=True)

import logging
import tqdm
import threading
from src.utils.distribution_utils import resolve_strategy, resolve_train_batch_sizes
print("[DEBUG:loaders.py] All imports complete!", flush=True)

logging.getLogger('nibabel').setLevel(logging.WARNING)

_NNUNET_SLICE_STEM_RE = re.compile(r"^(?P<volume_id>.+)_s(?P<slice_index>\d{4})$")

# A dictionary to map modality names to their processing functions
MODALITY_PROCESSORS = {
    'NCCT': process_ncct,
    'CTA': process_cta,
    'CBF': process_cbf,
    'CBV': process_cbv,
    'MTT': process_mtt,
    'TMAX': process_tmax
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
            if k == 'caseID':
                d[k] = v[0] if isinstance(v, list) else v
            elif isinstance(v, str) and len(v) > 0:
                d[k] = os.path.join(basedir, v)
            elif isinstance(v, list):
                d[k] = [os.path.join(basedir, iv) for iv in v]

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
    def __init__(self, directory, datalist_json, fold=0, transform=None, modalities=None, test_flag=False):
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.transform = transform
        self.modalities = modalities if modalities is not None else []
        self.base_modalities = sorted(list(set(mod.split('_')[0] for mod in self.modalities)))

        train_files, val_files = datafold_read(datalist=datalist_json, basedir=self.directory, fold=fold)
        self.database = val_files if test_flag else train_files

    def __len__(self):
        return len(self.database)

    def _process_modalities(self, data):
        """Process each modality based on its configuration."""
        processed_images = {}
        for modality_config in self.modalities:
            base_modality = modality_config.split('_')[0]
            raw_data = data[base_modality]
            
            # RAW mode: not implemented for 3D - use processors.py pipelines instead
            if modality_config.endswith('_RAW'):
                raise NotImplementedError(
                    f"RAW mode '{modality_config}' is not supported for ISLES24Dataset3D. "
                    "Use src/data/processors.py pipelines for 3D volume processing."
                )
            
            raw_np = raw_data.numpy()
            finite_mask = np.isfinite(raw_np)
            if not finite_mask.any():
                data_stats = {'min_val': 0.0, 'max_val': 0.0, 'mean': 0.0, 'std': 0.0}
            else:
                finite_vals = raw_np[finite_mask]
                data_stats = {
                    'min_val': float(np.min(finite_vals)),
                    'max_val': float(np.max(finite_vals)),
                    'mean': float(np.mean(finite_vals)),
                    'std': float(np.std(finite_vals)),
                }
            
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
        keys_to_load = self.base_modalities + ['label']
        for key in keys_to_load:
            if key not in filedict or not filedict[key]:
                continue
            
            filepath = filedict[key][0] if isinstance(filedict[key], list) else filedict[key]
            if os.path.exists(filepath):
                nib_img = nibabel.load(filepath)
                data[key] = torch.tensor(nib_img.get_fdata(), dtype=torch.float32)

        # Process modalities to get normalized channels
        processed_images = self._process_modalities(data)
        
        # Stack processed channels to form the final image
        image_channels = [processed_images[f"processed_{m}"] for m in self.modalities]
        image = torch.stack(image_channels, dim=0)

        label = data.get('label')
        if label is None: # Handle test sets with no labels
            label = torch.zeros_like(image_channels[0]).unsqueeze(0)
        else:
            label = label.unsqueeze(0)

        if self.transform:
            state = torch.get_rng_state()
            image = self.transform(image)
            torch.set_rng_state(state)
            label = self.transform(label)
            
        return image, label, filedict['caseID']


class ISLES24Dataset2D(torch.utils.data.Dataset):
    """
    Dataset class for ISLES24 that returns 2D slices from 3D volumes.
    Inspired by BRATSDataset3D and CustomDataset3D.
    """
    def __init__(self, directory, datalist_json, fold=0, transform=None, modalities=None, test_flag=False, image_size=32, use_caching=False, shared_cache=None, cache_lock=None, aug_cfg=None, is_training=False):
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.transform = transform
        self.modalities = modalities if modalities is not None else []
        self.base_modalities = sorted(list(set(mod.split('_')[0] for mod in self.modalities)))

        train_files, val_files = datafold_read(datalist=datalist_json, basedir=self.directory, fold=fold)
        self.database = val_files if test_flag else train_files
        
        self.all_slices = []

        self.image_size = image_size
        self.use_caching = use_caching
        self._cache_prefix = "ts" if test_flag else "tr"
        
        # NEW: Initialize augmentation pipeline
        self.aug_cfg = aug_cfg
        self.is_training = is_training
        self.augmentation = None
        
        if self.is_training and self.aug_cfg is not None:
            from src.data.augmentation import AugmentationPipeline2D
            self.augmentation = AugmentationPipeline2D(self.aug_cfg)
            print(f"Initialized augmentation pipeline for training dataset")
        
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
            base_modality = modality_config.split('_')[0]
            raw_data = data_slice[base_modality]
            
            # RAW mode: skip normalization, passthrough raw intensity values
            # Used for nnU-Net export where nnU-Net handles its own normalization
            if modality_config.endswith('_RAW'):
                processed_images[f"processed_{modality_config}"] = raw_data.float()
                continue
            
            raw_np = raw_data.numpy()
            finite_mask = np.isfinite(raw_np)
            if not finite_mask.any():
                data_stats = {'min_val': 0.0, 'max_val': 0.0, 'mean': 0.0, 'std': 0.0}
            else:
                finite_vals = raw_np[finite_mask]
                data_stats = {
                    'min_val': float(np.min(finite_vals)),
                    'max_val': float(np.max(finite_vals)),
                    'mean': float(np.mean(finite_vals)),
                    'std': float(np.std(finite_vals)),
                }

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
                        keys_to_load = self.base_modalities + ['label']
                        for key in keys_to_load:
                            if key not in filedict or not filedict[key]:
                                continue
                            
                            filepath = filedict[key][0] if isinstance(filedict[key], list) else filedict[key]
                            if os.path.exists(filepath):
                                nib_img = nibabel.load(filepath)
                                data[key] = torch.from_numpy(nib_img.get_fdata().astype(np.float32))
                        self.cache[cache_key] = data
            
            data_slice = {}
            for key in self.cache[cache_key]:
                data_slice[key] = self.cache[cache_key][key][..., slice_idx]
        else:
            # Load directly without caching
            data_slice = {}
            keys_to_load = self.base_modalities + ['label']
            for key in keys_to_load:
                if key not in filedict or not filedict[key]:
                    continue
                
                filepath = filedict[key][0] if isinstance(filedict[key], list) else filedict[key]
                if os.path.exists(filepath):
                    nib_img = nibabel.load(filepath)
                    vol_data = torch.from_numpy(nib_img.get_fdata().astype(np.float32))
                    data_slice[key] = vol_data[..., slice_idx]
        
        processed_images = self._process_modalities(data_slice)
        
        image_channels = [processed_images[f"processed_{m}"] for m in self.modalities]
        image = torch.stack(image_channels, dim=0)

        label = data_slice.get('label')
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
            data_dict = {'image': image, 'label': label}
            data_dict = self.augmentation(data_dict)
            image = data_dict['image']
            label = data_dict['label']

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

        if self.num_modalities <= 0:
            raise ValueError("ISLES24NNUNet2D requires a non-empty modalities list.")

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

        self.aug_cfg = aug_cfg
        self.is_training = bool(is_training)
        self.augmentation = None
        if self.is_training and self.aug_cfg is not None:
            from src.data.augmentation import AugmentationPipeline2D

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

    def _load_raw_sample(self, sample):
        image_tensors = []
        sample_shape = None

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
            image_tensors.append(torch.from_numpy(image_np))

        label_nib = nibabel.load(sample["label_path"])
        label_np = label_nib.get_fdata().astype(np.float32)
        label_np = _normalize_nnunet_slice_array(label_np, sample["label_path"])

        if sample_shape is None:
            raise RuntimeError(f"No image channels loaded for sample '{sample['slice_stem']}'.")
        if tuple(label_np.shape) != sample_shape:
            raise ValueError(
                f"Image/label geometry mismatch for '{sample['slice_stem']}'. "
                f"image={sample_shape}, label={tuple(label_np.shape)}."
            )

        image = torch.stack(image_tensors, dim=0).float()
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


def _is_set(value):
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
    kwargs = {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(persistent_workers) if int(num_workers) > 0 else False,
    }
    if int(num_workers) > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return kwargs


def validate_dataset_contract(cfg):
    """Validate explicit data contract before constructing datasets/dataloaders."""
    loader_mode = OmegaConf.select(cfg, "data_mode.loader_mode")
    dim = OmegaConf.select(cfg, "data_mode.dim")
    modalities = OmegaConf.select(cfg, "dataset.modalities")
    num_modalities = OmegaConf.select(cfg, "dataset.num_modalities")
    dataset_id = OmegaConf.select(cfg, "dataset.id", default=OmegaConf.select(cfg, "dataset.name"))

    if loader_mode not in {"online_slices_3d_to_2d", "nnunet_slices_2d", "full_volumes_3d"}:
        raise ValueError(
            "Invalid data_mode.loader_mode. Expected one of "
            "{online_slices_3d_to_2d, nnunet_slices_2d, full_volumes_3d}, "
            f"got: {loader_mode}"
        )

    if dataset_id != "isles24":
        raise ValueError(f"Only dataset.id/name='isles24' is currently supported, got: {dataset_id}")

    if not isinstance(modalities, (list, tuple)) or len(modalities) == 0:
        raise ValueError("dataset.modalities must be a non-empty list.")
    if int(num_modalities) != len(modalities):
        raise ValueError(
            "Global invariant violated: len(dataset.modalities) must equal dataset.num_modalities. "
            f"Got len(modalities)={len(modalities)}, num_modalities={num_modalities}."
        )

    runtime_required = [
        "data_runtime.train_batch_size",
        "data_runtime.test_batch_size",
        "data_runtime.num_train_workers",
        "data_runtime.num_valid_workers",
        "data_runtime.num_test_workers",
        "data_runtime.use_caching",
        "data_runtime.use_shared_cache",
        "data_runtime.train_prefetch_factor",
        "data_runtime.test_prefetch_factor",
    ]
    for key in runtime_required:
        value = OmegaConf.select(cfg, key)
        if value is None:
            raise ValueError(f"Missing required runtime key: {key}")

    if loader_mode == "online_slices_3d_to_2d":
        if dim != "2d":
            raise ValueError("online_slices_3d_to_2d requires data_mode.dim='2d'.")
        if not _is_set(OmegaConf.select(cfg, "data_io.paths.data_root")):
            raise ValueError("online_slices_3d_to_2d requires data_io.paths.data_root.")
        if not _is_set(OmegaConf.select(cfg, "data_io.paths.split_file")):
            raise ValueError("online_slices_3d_to_2d requires data_io.paths.split_file.")

    elif loader_mode == "nnunet_slices_2d":
        if dim != "2d":
            raise ValueError("nnunet_slices_2d requires data_mode.dim='2d'.")
        if not _is_set(OmegaConf.select(cfg, "data_io.paths.nnunet_root")):
            raise ValueError("nnunet_slices_2d requires data_io.paths.nnunet_root.")
        if not _is_set(OmegaConf.select(cfg, "dataset.nnunet.dataset_id")):
            raise ValueError("nnunet_slices_2d requires dataset.nnunet.dataset_id.")
        if not _is_set(OmegaConf.select(cfg, "dataset.nnunet.dataset_name")):
            raise ValueError("nnunet_slices_2d requires dataset.nnunet.dataset_name.")

    elif loader_mode == "full_volumes_3d":
        if dim != "3d":
            raise ValueError("full_volumes_3d requires data_mode.dim='3d'.")
        if not _is_set(OmegaConf.select(cfg, "data_io.paths.data_root")):
            raise ValueError("full_volumes_3d requires data_io.paths.data_root.")
        if not _is_set(OmegaConf.select(cfg, "data_io.paths.split_file")):
            raise ValueError("full_volumes_3d requires data_io.paths.split_file.")


def get_dataloaders(cfg):
    validate_dataset_contract(cfg)

    aug_cfg = cfg.augmentation if hasattr(cfg, 'augmentation') else None
    if aug_cfg is not None:
        spatial_enabled = aug_cfg.spatial.enabled
        intensity_enabled = aug_cfg.intensity.enabled
        if spatial_enabled or intensity_enabled:
            print(f"Augmentation enabled: spatial={spatial_enabled}, intensity={intensity_enabled}")
        else:
            print("Augmentation config present but all transforms disabled")
    else:
        print("No augmentation configured (using baseline)")

    loader_mode = cfg.data_mode.loader_mode
    data_root = cfg.data_io.paths.data_root
    split_file = cfg.data_io.paths.split_file

    if loader_mode == "nnunet_slices_2d":
        shared_cache = {} if cfg.data_runtime.use_shared_cache else None
        cache_lock = threading.Lock() if shared_cache else None

        train_dataset = ISLES24NNUNet2D(
            nnunet_root=cfg.data_io.paths.nnunet_root,
            dataset_id=cfg.dataset.nnunet.dataset_id,
            dataset_name=cfg.dataset.nnunet.dataset_name,
            modalities=cfg.dataset.modalities,
            test_flag=False,
            image_size=cfg.model.image_size,
            transform=None,
            use_caching=cfg.data_runtime.use_caching,
            shared_cache=shared_cache,
            cache_lock=cache_lock,
            aug_cfg=aug_cfg,
            is_training=True,
        )
        test_dataset = ISLES24NNUNet2D(
            nnunet_root=cfg.data_io.paths.nnunet_root,
            dataset_id=cfg.dataset.nnunet.dataset_id,
            dataset_name=cfg.dataset.nnunet.dataset_name,
            modalities=cfg.dataset.modalities,
            test_flag=True,
            image_size=cfg.model.image_size,
            transform=None,
            use_caching=cfg.data_runtime.use_caching,
            shared_cache=shared_cache,
            cache_lock=cache_lock,
            aug_cfg=None,
            is_training=False,
        )

    elif loader_mode == "online_slices_3d_to_2d":
        shared_cache = {} if cfg.data_runtime.use_shared_cache else None
        cache_lock = threading.Lock() if shared_cache else None

        train_dataset = ISLES24OnlineProc2D(
            directory=data_root,
            datalist_json=split_file,
            fold=cfg.dataset.fold,
            transform=None,
            modalities=cfg.dataset.modalities,
            test_flag=False,
            image_size=cfg.model.image_size,
            use_caching=cfg.data_runtime.use_caching,
            shared_cache=shared_cache,
            cache_lock=cache_lock,
            aug_cfg=aug_cfg,
            is_training=True,
        )
        test_dataset = ISLES24OnlineProc2D(
            directory=data_root,
            datalist_json=split_file,
            fold=cfg.dataset.fold,
            transform=None,
            modalities=cfg.dataset.modalities,
            test_flag=True,
            image_size=cfg.model.image_size,
            use_caching=cfg.data_runtime.use_caching,
            shared_cache=shared_cache,
            cache_lock=cache_lock,
            aug_cfg=None,
            is_training=False,
        )
    elif loader_mode == "full_volumes_3d":
        train_dataset = ISLES24Dataset3D(
            directory=data_root,
            datalist_json=split_file,
            fold=cfg.dataset.fold,
            transform=None,
            modalities=cfg.dataset.modalities,
            test_flag=False,
        )
        test_dataset = ISLES24Dataset3D(
            directory=data_root,
            datalist_json=split_file,
            fold=cfg.dataset.fold,
            transform=None,
            modalities=cfg.dataset.modalities,
            test_flag=True,
        )
    else:
        raise ValueError(f"Unsupported loader_mode: {loader_mode}")

    strategy = resolve_strategy(cfg)
    global_train_batch_size, per_rank_train_batch_size = resolve_train_batch_sizes(
        int(cfg.data_runtime.train_batch_size),
        strategy=strategy,
    )
    print(
        f"Train batch semantics: global={global_train_batch_size}, "
        f"per_rank={per_rank_train_batch_size}, strategy={strategy}"
    )

    train_sampler = None
    is_ddp = strategy == "ddp"
    if is_ddp:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError(
                "distribution=ddp requires an initialized torch.distributed process group "
                "before dataloader construction."
            )
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=True,
        )

    train_kwargs = _build_loader_kwargs(
        num_workers=int(cfg.data_runtime.num_train_workers),
        pin_memory=bool(OmegaConf.select(cfg, "data_runtime.pin_memory.train", default=False)),
        persistent_workers=bool(OmegaConf.select(cfg, "data_runtime.persistent_workers.train", default=False)),
        prefetch_factor=OmegaConf.select(cfg, "data_runtime.train_prefetch_factor", default=None),
    )
    val_kwargs = _build_loader_kwargs(
        num_workers=int(cfg.data_runtime.num_valid_workers),
        pin_memory=bool(OmegaConf.select(cfg, "data_runtime.pin_memory.val", default=False)),
        persistent_workers=bool(OmegaConf.select(cfg, "data_runtime.persistent_workers.val", default=False)),
        prefetch_factor=OmegaConf.select(cfg, "data_runtime.test_prefetch_factor", default=None),
    )
    sample_kwargs = _build_loader_kwargs(
        num_workers=int(cfg.data_runtime.num_test_workers),
        pin_memory=bool(OmegaConf.select(cfg, "data_runtime.pin_memory.test", default=False)),
        persistent_workers=bool(OmegaConf.select(cfg, "data_runtime.persistent_workers.test", default=False)),
        prefetch_factor=OmegaConf.select(cfg, "data_runtime.test_prefetch_factor", default=None),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=per_rank_train_batch_size,
        shuffle=not is_ddp,
        sampler=train_sampler,
        **train_kwargs,
    )
    val_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.validation.val_batch_size,
        shuffle=False,
        **val_kwargs,
    )
    sample_dataloader = DataLoader(
        test_dataset,
        batch_size=int(cfg.data_runtime.test_batch_size),
        shuffle=True,
        **sample_kwargs,
    )

    return {
        'train': train_dataloader,
        'val': val_dataloader,
        'sample': sample_dataloader
    }
