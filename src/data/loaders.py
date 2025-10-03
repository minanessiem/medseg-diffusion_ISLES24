import os
import glob
# import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

import json
import os
import torch
import numpy as np
import nibabel
from monai.transforms import Resize
from src.data.processors import get_image_transform, get_mask_transform, get_joint_transform

from src.data.modalities import get_modality_params
from src.data.modalities import process_cbf
from src.data.modalities import process_cbv
from src.data.modalities import process_cta
from src.data.modalities import process_mtt
from src.data.modalities import process_ncct
from src.data.modalities import process_tmax

import logging

import tqdm

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
    def __init__(self, directory, datalist_json, fold=0, transform=None, modalities=None, test_flag=False, image_size=32):
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.transform = transform
        self.modalities = modalities if modalities is not None else []
        self.base_modalities = sorted(list(set(mod.split('_')[0] for mod in self.modalities)))

        train_files, val_files = datafold_read(datalist=datalist_json, basedir=self.directory, fold=fold)
        self.database = val_files if test_flag else train_files
        
        self.all_slices = []

        self.image_size = image_size
        print("Pre-calculating dataset size...")
        for case_idx, filedict in tqdm.tqdm(enumerate(self.database), total=len(self.database)):
            # Use the first modality to determine the number of slices
            first_mod_key = self.base_modalities[0]
            filepath = filedict[first_mod_key][0] if isinstance(filedict[first_mod_key], list) else filedict[first_mod_key]
            if os.path.exists(filepath):
                num_slices = nibabel.load(filepath).shape[-1]
                self.all_slices.extend([(case_idx, slice_idx) for slice_idx in range(num_slices)])
        
        self.cache = {}  # Cache for preloaded 3D volumes per case_idx
        

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
        
        if case_idx not in self.cache:
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

        if self.transform:
            state = torch.get_rng_state()
            image = self.transform(image)
            torch.set_rng_state(state)
            label = self.transform(label)
            
        virtual_path = f"{filedict['caseID']}_slice{slice_idx}"
        return image, label, virtual_path 


def pos_neg_diagnosis(mask_path):
    """
    Determines whether a mask image indicates the presence of cancer.

    Args:
        mask_path (str): Path to the mask image file.

    Returns:
        int: 1 if the mask contains positive values (cancer detected), 0 otherwise.
    """
    try:
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Failed to load image at path: {mask_path}")
        return int(np.max(mask) > 0)
    except Exception as e:
        raise RuntimeError(f"Error processing mask at {mask_path}: {e}")

def load_mri_df(mri_scans_path):
    """
    Loads MRI scan data into a DataFrame, including cancer diagnosis based on mask files.

    Args:
        mri_scans_path (str): Path to the root directory containing subdirectories of MRI scans.

    Returns:
        pd.DataFrame: A DataFrame with columns:
                      - 'patient_id': Patient identifier derived from subdirectory names.
                      - 'image_path': Path to the MRI image files.
                      - 'mask_path': Path to the corresponding mask image files.
                      - 'has_cancer': 1 if the mask indicates cancer, 0 otherwise.
    """
    data_records = []
    
    # Iterate through subdirectories
    for sub_dir_path in glob.glob(os.path.join(mri_scans_path, "*/")):
        dir_name = os.path.basename(os.path.normpath(sub_dir_path))
        
        try:
            # Collect image and mask files
            image_files = [f for f in os.listdir(sub_dir_path) if not f.endswith('mask.tif')]
            mask_files = [f for f in os.listdir(sub_dir_path) if f.endswith('mask.tif')]
            
            # Match image and mask pairs
            for image_file, mask_file in zip(sorted(image_files), sorted(mask_files)):
                image_path = os.path.join(sub_dir_path, image_file)
                mask_path = os.path.join(sub_dir_path, mask_file)
                data_records.append([dir_name, image_path, mask_path])
        except Exception as e:
            # Log the specific directory that caused an issue
            print(f"Error processing directory '{sub_dir_path}': {e}")

    # Create a DataFrame and compute cancer diagnosis
    mri_df = pd.DataFrame(data_records, columns=['patient_id', 'image_path', 'mask_path'])
    mri_df['has_cancer'] = mri_df['mask_path'].apply(pos_neg_diagnosis)
    
    return mri_df

def split_dataset(mri_df, cfg):
    """
    Splits the dataset into training and testing sets.

    Args:
        mri_df (pd.DataFrame): The dataframe containing MRI data.
        cfg (DictConfig): Hydra configuration with split params (e.g., test_size, random_seed).

    Returns:
        tuple: A tuple containing the training and testing dataframes (train_df, test_df).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        mri_df[['image_path']],
        mri_df[['mask_path', 'has_cancer']],
        test_size=cfg.dataset.test_size,
        random_state=cfg.random_seed,
        stratify=mri_df['has_cancer'],
    )

    train_df = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    test_df = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)
    return train_df, test_df

class BrainMRIDataset(Dataset):
    def __init__(self, dataframe, cfg):
        self.dataframe = dataframe
        self.image_transform = get_image_transform(cfg)
        self.mask_transform = get_mask_transform(cfg)
        self.joint_transform = get_joint_transform(cfg)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image = self.get_unprocessed_image(idx)
        mask = self.get_unprocessed_mask(idx)
        label = int(self.dataframe['has_cancer'][idx])
        transformed = self.joint_transform(image=image, mask=mask)

        image = transformed['image']
        mask = transformed['mask']

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        return image, mask, torch.tensor(label).int()

    def get_unprocessed_mask(self, idx):
        mask_path = self.dataframe['mask_path'][idx]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return mask

    def get_unprocessed_image(self, idx):
        image_path = self.dataframe['image_path'][idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

def get_dataloaders(cfg):
    """
    Factory to load MRI dataframe, split, and create train/test dataloaders.

    Args:
        cfg (DictConfig): Hydra configuration object.

    Returns:
        train_dataloader, test_dataloader
    """
    if cfg.dataset.name == 'isles24':
        train_dataset = ISLES24Dataset2D(
            directory=cfg.dataset.dir,
            datalist_json=cfg.dataset.json_list,
            fold=cfg.dataset.fold,
            transform=None,
            modalities=cfg.dataset.modalities,
            test_flag=False,
            image_size=cfg.model.image_size
        )
        test_dataset = ISLES24Dataset2D(
            directory=cfg.dataset.dir,
            datalist_json=cfg.dataset.json_list,
            fold=cfg.dataset.fold,
            transform=None,
            modalities=cfg.dataset.modalities,
            test_flag=True,
            image_size=cfg.model.image_size
        )
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=cfg.dataset.train_batch_size, 
            shuffle=True,
            num_workers=cfg.dataset.num_workers,
            pin_memory=True,
            persistent_workers=True if cfg.dataset.num_workers > 0 else False
        )
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=cfg.dataset.test_batch_size, 
            shuffle=False,
            num_workers=cfg.dataset.num_workers,
            pin_memory=True,
            persistent_workers=True if cfg.dataset.num_workers > 0 else False
        )
        return train_dataloader, test_dataloader
    else:
        mri_df = load_mri_df(cfg.dataset.dir)
        train_df, test_df = split_dataset(mri_df, cfg)

        train_dataset = BrainMRIDataset(train_df, cfg)
        test_dataset = BrainMRIDataset(test_df, cfg)

        train_dataloader = DataLoader(train_dataset, batch_size=cfg.dataset.train_batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=cfg.dataset.test_batch_size, shuffle=False)

        return train_dataloader, test_dataloader
