import sys
print("[DEBUG:loaders.py] Starting imports...", flush=True)

import os
import glob
print("[DEBUG:loaders.py] os, glob done", flush=True)

import numpy as np
import pandas as pd
print("[DEBUG:loaders.py] numpy, pandas done", flush=True)

from sklearn.model_selection import train_test_split
print("[DEBUG:loaders.py] sklearn done", flush=True)

import torch
from torch.utils.data import Dataset, DataLoader
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
print("[DEBUG:loaders.py] All imports complete!", flush=True)

logging.getLogger('nibabel').setLevel(logging.WARNING)

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
        
        if self.use_caching:
            if case_idx not in self.cache:
                with self.cache_lock:
                    if case_idx not in self.cache:  # Double-check after acquiring lock
                        data = {}
                        keys_to_load = self.base_modalities + ['label']
                        for key in keys_to_load:
                            if key not in filedict or not filedict[key]:
                                continue
                            
                            filepath = filedict[key][0] if isinstance(filedict[key], list) else filedict[key]
                            if os.path.exists(filepath):
                                nib_img = nibabel.load(filepath)
                                data[key] = torch.from_numpy(nib_img.get_fdata().astype(np.float32))
                        self.cache[case_idx] = data
            
            data_slice = {}
            for key in self.cache[case_idx]:
                data_slice[key] = self.cache[case_idx][key][..., slice_idx]
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

def get_dataloaders(cfg):
    OmegaConf.set_struct(cfg, False)
    # Temporary aliases for config transition
    cfg.dataset.dir = cfg.environment.dataset.dir
    cfg.dataset.json_list = cfg.environment.dataset.json_list
    cfg.dataset.num_train_workers = cfg.environment.dataset.num_train_workers
    cfg.dataset.num_valid_workers = cfg.environment.dataset.num_valid_workers
    cfg.dataset.num_test_workers = cfg.environment.dataset.num_test_workers
    cfg.dataset.train_batch_size = cfg.environment.dataset.train_batch_size
    cfg.dataset.test_batch_size = cfg.environment.dataset.test_batch_size
    OmegaConf.set_struct(cfg, True)

    # NEW: Extract augmentation config from top-level
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

    if cfg.dataset.name == 'isles24':
        shared_cache = {} if cfg.dataset.use_shared_cache else None  # Optional config flag for easy toggling
        cache_lock = threading.Lock() if shared_cache else None
        
        train_dataset = ISLES24Dataset2D(
            directory=cfg.dataset.dir,
            datalist_json=cfg.dataset.json_list,
            fold=cfg.dataset.fold,
            transform=None,
            modalities=cfg.dataset.modalities,
            test_flag=False,
            image_size=cfg.model.image_size,
            use_caching=cfg.dataset.use_caching,
            shared_cache=shared_cache,
            cache_lock=cache_lock,
            aug_cfg=aug_cfg,      # NEW
            is_training=True       # NEW
        )
        test_dataset = ISLES24Dataset2D(
            directory=cfg.dataset.dir,
            datalist_json=cfg.dataset.json_list,
            fold=cfg.dataset.fold,
            transform=None,
            modalities=cfg.dataset.modalities,
            test_flag=True,
            image_size=cfg.model.image_size,
            use_caching=cfg.dataset.use_caching,
            shared_cache=shared_cache,
            cache_lock=cache_lock,
            aug_cfg=None,          # NEW: explicitly None for validation
            is_training=False      # NEW
        )
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=cfg.dataset.train_batch_size, 
            shuffle=True,
            num_workers=cfg.dataset.num_train_workers,
            pin_memory=True,
            persistent_workers=False, # True if cfg.dataset.num_workers > 0 else False,
            prefetch_factor=cfg.dataset.train_prefetch_factor
        )
        val_dataloader = DataLoader(
            test_dataset, 
            batch_size=cfg.validation.val_batch_size, 
            shuffle=False,
            num_workers=cfg.dataset.num_valid_workers,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=cfg.dataset.test_prefetch_factor
        )
        sample_dataloader = DataLoader(
            test_dataset, 
            batch_size=cfg.dataset.test_batch_size, 
            shuffle=True,
            num_workers=cfg.dataset.num_test_workers,
            pin_memory=False,
            persistent_workers=False,
            # prefetch_factor=cfg.dataset.test_prefetch_factor
        )
        return {
            'train': train_dataloader,
            'val': val_dataloader,
            'sample': sample_dataloader
        }
    else:
        raise ValueError(f"Dataset {cfg.dataset.name} not implemented")
