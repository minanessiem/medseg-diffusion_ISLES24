import os
import glob
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

# Assuming processors.py has get_image_transform, etc.; import them
from src.data.processors import get_image_transform, get_mask_transform, get_joint_transform

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
    mri_df = load_mri_df(cfg.dataset.dir)
    train_df, test_df = split_dataset(mri_df, cfg)

    train_dataset = BrainMRIDataset(train_df, cfg)
    test_dataset = BrainMRIDataset(test_df, cfg)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.dataset.train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.dataset.test_batch_size, shuffle=False)

    return train_dataloader, test_dataloader