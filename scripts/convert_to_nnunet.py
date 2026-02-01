#!/usr/bin/env python3
"""
Convert ISLES24 dataset to nnU-Net v2 format using our preprocessing pipeline.

Uses existing config infrastructure and data loading structure.
Outputs preprocessed 2D slices in nnU-Net format for benchmarking.

Usage:
    # Test mode (default - processes limited slices)
    python scripts/convert_to_nnunet.py --config-name=convert_nnunet
    
    # Full export
    python scripts/convert_to_nnunet.py --config-name=convert_nnunet nnunet.test=false
    
    # Override output location
    python scripts/convert_to_nnunet.py --config-name=convert_nnunet \
        nnunet.output_dir=/mnt/data/nnUNet_raw
    
    # Different fold
    python scripts/convert_to_nnunet.py --config-name=convert_nnunet \
        dataset.fold=2
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Import MONAI first (before CUDA context creation)
try:
    from monai.transforms import Resize, ScaleIntensityRange
except ImportError:
    pass

import hydra
from omegaconf import DictConfig, OmegaConf
import nibabel as nib
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

from src.utils.train_utils import setup_seeds
from src.data.loaders import get_dataloaders


def setup_export_config_aliases(cfg: DictConfig) -> DictConfig:
    """
    Set up minimal config aliases needed for data loading during export.
    
    Only sets dataset-related aliases required by get_dataloaders().
    Does not require training config.
    """
    OmegaConf.set_struct(cfg, False)
    
    # Dataset aliases (needed by get_dataloaders)
    cfg.dataset.dir = cfg.environment.dataset.dir
    cfg.dataset.json_list = cfg.environment.dataset.json_list
    cfg.dataset.num_train_workers = cfg.environment.dataset.num_train_workers
    cfg.dataset.num_valid_workers = cfg.environment.dataset.num_valid_workers
    cfg.dataset.num_test_workers = cfg.environment.dataset.num_test_workers
    cfg.dataset.train_batch_size = cfg.environment.dataset.train_batch_size
    cfg.dataset.test_batch_size = cfg.environment.dataset.test_batch_size
    
    OmegaConf.set_struct(cfg, True)
    return cfg


def export_dataset(
    dataset, 
    images_dir: Path, 
    labels_dir: Path, 
    num_channels: int, 
    desc: str,
    max_slices: int = None
) -> set:
    """
    Export a dataset to nnU-Net format by iterating deterministically by index.
    
    Args:
        dataset: ISLES24Dataset2D instance
        images_dir: Output directory for images
        labels_dir: Output directory for labels
        num_channels: Number of modality channels
        desc: Progress bar description
        max_slices: Maximum number of slices to export (None for all)
    
    Returns:
        Set of exported case identifiers
    """
    case_ids = set()
    
    # Determine number of slices to process
    total_slices = len(dataset)
    if max_slices is not None:
        num_to_process = min(max_slices, total_slices)
    else:
        num_to_process = total_slices
    
    # Deterministic iteration by index (bypasses any dataloader shuffle)
    for idx in tqdm(range(num_to_process), desc=desc):
        image, label, virtual_path = dataset[idx]
        # image: [C, H, W], label: [1, H, W]
        
        # Parse case_id and slice_idx from virtual_path
        # Format from ISLES24Dataset2D: "{caseID}_slice{slice_idx}"
        case_id, slice_part = virtual_path.rsplit("_slice", 1)
        slice_idx = int(slice_part)
        
        # Create case identifier for nnU-Net
        safe_case_id = f"{case_id}_s{slice_idx:04d}"
        case_ids.add(safe_case_id)
        
        # Save each channel as separate file
        for ch_idx in range(num_channels):
            ch_data = image[ch_idx].numpy()  # [H, W]
            
            # Create 2D NIfTI (shape [H, W, 1] for 2D)
            nii_data = ch_data[..., np.newaxis].astype(np.float32)
            nii_img = nib.Nifti1Image(nii_data, affine=np.eye(4))
            
            filename = f"{safe_case_id}_{ch_idx:04d}.nii.gz"
            nib.save(nii_img, images_dir / filename)
        
        # Save label (as uint8 for segmentation)
        label_data = label[0].numpy()  # [H, W]
        label_nii = label_data[..., np.newaxis].astype(np.uint8)
        label_img = nib.Nifti1Image(label_nii, affine=np.eye(4))
        nib.save(label_img, labels_dir / f"{safe_case_id}.nii.gz")
    
    return case_ids


@hydra.main(config_path="../configs", config_name="convert_nnunet", version_base=None)
def main(cfg: DictConfig):
    """
    Main conversion entry point.
    
    Reuses existing config infrastructure and data loading.
    All settings come from config - no defaults in code.
    """
    # === 1. Setup config (minimal aliases for data loading) ===
    cfg = setup_export_config_aliases(cfg)
    setup_seeds(cfg)
    
    # === 2. Build output paths from config ===
    # NOTE: In nnU-Net, ALL data for training+validation goes into imagesTr/labelsTr
    # The train/val split is defined by splits_final.json, NOT by folder location
    # imagesTs is ONLY for held-out test data that won't be used during training
    output_base = Path(cfg.nnunet.output_dir)
    dataset_folder = output_base / f"Dataset{cfg.nnunet.dataset_id}_{cfg.nnunet.dataset_name}"
    
    imagesTr = dataset_folder / "imagesTr"
    labelsTr = dataset_folder / "labelsTr"
    
    for d in [imagesTr, labelsTr]:
        d.mkdir(parents=True, exist_ok=True)
    
    # === 3. Determine test mode settings ===
    is_test_mode = cfg.nnunet.test
    max_slices = cfg.nnunet.test_max_slices if is_test_mode else None
    
    # === 4. Print configuration summary ===
    print(f"\n{'='*60}")
    print(f"nnU-Net Dataset Conversion")
    print(f"{'='*60}")
    print(f"Mode: {'TEST (limited slices)' if is_test_mode else 'FULL EXPORT'}")
    if is_test_mode:
        print(f"Max slices per split: {max_slices}")
    print(f"Source dataset: {cfg.dataset.name}")
    print(f"Modalities: {list(cfg.dataset.modalities)}")
    print(f"Image size: {cfg.model.image_size}")
    print(f"Validation fold: {cfg.dataset.fold}")
    print(f"Augmentation: {cfg.augmentation._name_}")
    print(f"Output: {dataset_folder}")
    print(f"{'='*60}\n")
    
    # === 5. Get dataloaders, extract underlying datasets ===
    dataloaders = get_dataloaders(cfg)
    
    # Access underlying datasets (bypasses dataloader shuffle)
    train_dataset = dataloaders['train'].dataset
    val_dataset = dataloaders['val'].dataset
    
    num_channels = len(cfg.dataset.modalities)
    
    print(f"Total training slices: {len(train_dataset)}")
    print(f"Total validation slices: {len(val_dataset)}")
    print(f"Channels per slice: {num_channels}")
    if is_test_mode:
        print(f"Will export: {min(max_slices, len(train_dataset))} train, {min(max_slices, len(val_dataset))} val")
    print()
    
    # === 6. Export datasets (deterministic iteration by index) ===
    # Both train and val go into imagesTr/labelsTr - split is defined by splits_final.json
    train_case_ids = export_dataset(
        train_dataset, imagesTr, labelsTr, 
        num_channels, "Exporting training set",
        max_slices=max_slices
    )
    val_case_ids = export_dataset(
        val_dataset, imagesTr, labelsTr,  # Same folder as train!
        num_channels, "Exporting validation set",
        max_slices=max_slices
    )
    
    # === 7. Generate dataset.json ===
    channel_names = {
        str(i): cfg.dataset.modalities[i] 
        for i in range(num_channels)
    }
    
    # numTraining = total cases in imagesTr (both train and val)
    total_cases = len(train_case_ids) + len(val_case_ids)
    
    dataset_json = {
        "channel_names": channel_names,
        "labels": {
            "background": 0,
            "lesion": 1
        },
        "numTraining": total_cases,
        "file_ending": ".nii.gz",
        "description": f"ISLES24 2D slices preprocessed with MedSegDiff pipeline",
        "reference": "Converted from MedSegDiff preprocessing",
        "source_config": {
            "dataset": cfg.dataset.name,
            "modalities": list(cfg.dataset.modalities),
            "image_size": cfg.model.image_size,
            "fold": cfg.dataset.fold,
            "test_mode": is_test_mode
        }
    }
    
    with open(dataset_folder / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)
    
    # === 8. Generate splits_final.json ===
    splits = [{
        "train": sorted(list(train_case_ids)),
        "val": sorted(list(val_case_ids))
    }]
    
    with open(dataset_folder / "splits_final.json", "w") as f:
        json.dump(splits, f, indent=2)
    
    # === 9. Summary ===
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"{'='*60}")
    print(f"Training slices: {len(train_case_ids)}")
    print(f"Validation slices: {len(val_case_ids)}")
    print(f"Total in imagesTr: {total_cases}")
    if is_test_mode:
        print(f"\n⚠️  TEST MODE: Only exported {max_slices} slices per split")
        print(f"   Run with nnunet.test=false for full export")
    print(f"\nOutput structure:")
    print(f"  {dataset_folder}/")
    print(f"  ├── dataset.json")
    print(f"  ├── splits_final.json  (defines train/val split)")
    print(f"  ├── imagesTr/  ({total_cases * num_channels} files - both train & val)")
    print(f"  └── labelsTr/  ({total_cases} files - both train & val)")
    print(f"\nNext steps for nnU-Net:")
    print(f"  1. export nnUNet_raw={output_base}")
    print(f"  2. nnUNetv2_plan_and_preprocess -d {cfg.nnunet.dataset_id}")
    print(f"  3. Copy splits_final.json to nnUNet_preprocessed/Dataset{cfg.nnunet.dataset_id}_{cfg.nnunet.dataset_name}/")
    print(f"  4. nnUNetv2_train {cfg.nnunet.dataset_id} 2d 0 --npz")


if __name__ == "__main__":
    main()

