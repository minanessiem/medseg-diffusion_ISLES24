#!/usr/bin/env python3
"""
Convert ISLES24 2D dataset to nnU-Net v2 format using our preprocessing pipeline.

This script is specifically designed for ISLES24Dataset2D, which provides
preprocessed 2D axial slices. It exports these slices to nnU-Net format
for benchmarking against our diffusion-based segmentation method.

Dataset-specific assumptions:
    - Uses ISLES24Dataset2D via get_dataloaders()
    - Parses slice paths in format "{caseID}_slice{idx}"
    - Outputs 2D NIfTI files with shape [H, W, 1]

For 3D volumes or other datasets, create a new converter script
(e.g., convert_isles24_3d_dataset_to_nnunet.py).

Usage:
    # Local environment - test mode (default - processes limited slices)
    python3 -m scripts.nnunet.convert_isles24_2d_dataset_to_nnunet \
        --config-name=convert_nnunet_local
    
    # Cluster environment
    python3 -m scripts.nnunet.convert_isles24_2d_dataset_to_nnunet \
        --config-name=convert_nnunet_cluster
    
    # Full export
    python3 -m scripts.nnunet.convert_isles24_2d_dataset_to_nnunet \
        --config-name=convert_nnunet_local nnunet.test=false
    
    # Override output location
    python3 -m scripts.nnunet.convert_isles24_2d_dataset_to_nnunet \
        --config-name=convert_nnunet_local \
        nnunet.output_dir=/mnt/data/nnUNet_raw
    
    # Different fold
    python3 -m scripts.nnunet.convert_isles24_2d_dataset_to_nnunet \
        --config-name=convert_nnunet_local \
        dataset.fold=2
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Import MONAI first (before CUDA context creation)
try:
    from monai.transforms import Resize, ScaleIntensityRange
except ImportError:
    pass

import gc
import shutil
import hydra
from omegaconf import DictConfig, OmegaConf
import nibabel as nib
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

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


def clear_directory(directory: Path, description: str = "") -> int:
    """
    Remove all files from a directory, keeping the directory itself.
    
    Args:
        directory: Path to directory to clear
        description: Human-readable description for logging
    
    Returns:
        Number of files removed
    """
    if not directory.exists():
        return 0
    
    files = list(directory.glob("*"))
    count = len(files)
    
    if count > 0:
        # Remove and recreate for speed (faster than individual deletes)
        shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
        label = f" ({description})" if description else ""
        print(f"  Cleared {count} stale files from {directory.name}/{label}")
    
    return count


def count_files_in_directory(directory: Path, pattern: str = "*.nii.gz") -> int:
    """
    Count files matching a pattern in a directory.
    
    Used to determine numTraining from existing files on disk
    when only exporting the test split (resume scenario).
    
    Args:
        directory: Path to directory
        pattern: Glob pattern to match
    
    Returns:
        Number of matching files
    """
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def process_single_slice(idx: int, dataset, images_dir: Path, labels_dir: Path, 
                         num_channels: int) -> str:
    """
    Process and save a single slice to nnU-Net format.
    
    Args:
        idx: Slice index in dataset
        dataset: ISLES24Dataset2D instance
        images_dir: Output directory for images
        labels_dir: Output directory for labels
        num_channels: Number of modality channels
    
    Returns:
        Case identifier string
    """
    image, label, virtual_path = dataset[idx]
    # image: [C, H, W], label: [1, H, W]
    
    # Parse case_id and slice_idx from virtual_path
    # Format from ISLES24Dataset2D: "{caseID}_slice{slice_idx}"
    case_id, slice_part = virtual_path.rsplit("_slice", 1)
    slice_idx = int(slice_part)
    
    # Create case identifier for nnU-Net
    safe_case_id = f"{case_id}_s{slice_idx:04d}"
    
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
    
    return safe_case_id


def export_dataset(
    dataset, 
    images_dir: Path, 
    labels_dir: Path, 
    num_channels: int, 
    desc: str,
    max_slices: int = None
) -> set:
    """
    Export a dataset to nnU-Net format sequentially.
    
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
    num_to_process = min(max_slices, total_slices) if max_slices else total_slices
    
    # Sequential iteration by index
    for idx in tqdm(range(num_to_process), desc=desc):
        case_id = process_single_slice(idx, dataset, images_dir, labels_dir, num_channels)
        case_ids.add(case_id)
    
    return case_ids


def export_dataset_parallel(
    dataset, 
    images_dir: Path, 
    labels_dir: Path, 
    num_channels: int, 
    desc: str,
    max_slices: int = None,
    num_workers: int = 32
) -> set:
    """
    Export a dataset to nnU-Net format using parallel threads.
    
    Uses ThreadPoolExecutor for parallel processing. GIL is released
    during I/O operations (nibabel read/write), allowing good parallelism.
    
    Args:
        dataset: ISLES24Dataset2D instance
        images_dir: Output directory for images
        labels_dir: Output directory for labels
        num_channels: Number of modality channels
        desc: Progress bar description
        max_slices: Maximum number of slices to export (None for all)
        num_workers: Number of parallel threads
    
    Returns:
        Set of exported case identifiers
    """
    # Determine number of slices to process
    total_slices = len(dataset)
    num_to_process = min(max_slices, total_slices) if max_slices else total_slices
    
    case_ids = set()
    
    # Create partial function with fixed arguments
    process_fn = partial(
        process_single_slice,
        dataset=dataset,
        images_dir=images_dir,
        labels_dir=labels_dir,
        num_channels=num_channels
    )
    
    # Process slices in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_fn, idx): idx for idx in range(num_to_process)}
        
        # Collect results with progress bar
        for future in tqdm(as_completed(futures), total=num_to_process, desc=desc):
            case_id = future.result()
            case_ids.add(case_id)
    
    return case_ids


@hydra.main(config_path="../../configs", config_name="convert_nnunet_local", version_base=None)
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
    # Clean train/test split:
    # - imagesTr/labelsTr = Training data ONLY
    # - imagesTs/labelsTs = Validation/test data ONLY (configured fold)
    output_base = Path(cfg.nnunet.output_dir)
    dataset_folder = output_base / f"Dataset{cfg.nnunet.dataset_id}_{cfg.nnunet.dataset_name}"
    
    imagesTr = dataset_folder / "imagesTr"
    labelsTr = dataset_folder / "labelsTr"
    imagesTs = dataset_folder / "imagesTs"
    labelsTs = dataset_folder / "labelsTs"
    
    for d in [imagesTr, labelsTr, imagesTs, labelsTs]:
        d.mkdir(parents=True, exist_ok=True)
    
    # === 3. Determine test mode settings ===
    is_test_mode = cfg.nnunet.test
    max_slices = cfg.nnunet.test_max_slices if is_test_mode else None
    
    # === 4. Determine parallel settings ===
    use_parallel = cfg.nnunet.parallel.enabled
    num_workers = cfg.nnunet.parallel.num_workers
    
    # === 5. Determine which splits to export ===
    export_train = cfg.nnunet.export_train
    export_test = cfg.nnunet.export_test
    
    if not export_train and not export_test:
        raise ValueError("At least one of export_train or export_test must be True")
    
    # === 6. Print configuration summary ===
    print(f"\n{'='*60}")
    print(f"ISLES24 2D Dataset → nnU-Net Conversion")
    print(f"{'='*60}")
    print(f"Mode: {'TEST (limited slices)' if is_test_mode else 'FULL EXPORT'}")
    if is_test_mode:
        print(f"Max slices per split: {max_slices}")
    print(f"Export splits: {'train' if export_train else ''}{' + ' if export_train and export_test else ''}{'test' if export_test else ''}")
    print(f"Parallel: {'enabled (' + str(num_workers) + ' threads)' if use_parallel else 'disabled'}")
    print(f"Source dataset: {cfg.dataset.name}")
    print(f"Modalities: {list(cfg.dataset.modalities)}")
    print(f"Image size: {cfg.model.image_size}")
    print(f"Validation fold: {cfg.dataset.fold}")
    print(f"Augmentation: {cfg.augmentation._name_}")
    print(f"Output: {dataset_folder}")
    print(f"{'='*60}\n")
    
    # === 7. Get dataloaders, extract underlying datasets ===
    dataloaders = get_dataloaders(cfg)
    
    # Access underlying datasets (bypasses dataloader shuffle)
    train_dataset = dataloaders['train'].dataset
    val_dataset = dataloaders['val'].dataset
    
    num_channels = len(cfg.dataset.modalities)
    
    print(f"Total training slices: {len(train_dataset)}")
    print(f"Total validation slices: {len(val_dataset)}")
    print(f"Channels per slice: {num_channels}")
    if is_test_mode:
        if export_train:
            print(f"Will export train: {min(max_slices, len(train_dataset))} slices")
        if export_test:
            print(f"Will export test: {min(max_slices, len(val_dataset))} slices")
    print()
    
    # === 8. Export datasets ===
    # Clean separation: train → imagesTr/labelsTr, val → imagesTs/labelsTs
    train_case_ids = set()
    val_case_ids = set()
    
    if export_train:
        # Clear stale files from previous runs
        clear_directory(imagesTr, "training images")
        clear_directory(labelsTr, "training labels")
        
        if use_parallel:
            train_case_ids = export_dataset_parallel(
                train_dataset, imagesTr, labelsTr, 
                num_channels, "Exporting training set",
                max_slices=max_slices,
                num_workers=num_workers
            )
        else:
            train_case_ids = export_dataset(
                train_dataset, imagesTr, labelsTr, 
                num_channels, "Exporting training set",
                max_slices=max_slices
            )
    else:
        print("Skipping training set export (export_train=false)")
    
    # === 8b. Memory cleanup between phases ===
    # Free training data references before starting validation export
    # to avoid accumulating memory from both phases simultaneously
    del train_dataset
    del dataloaders
    gc.collect()
    
    if export_test:
        # Clear stale files from previous runs
        clear_directory(imagesTs, "test images")
        clear_directory(labelsTs, "test labels")
        
        if use_parallel:
            val_case_ids = export_dataset_parallel(
                val_dataset, imagesTs, labelsTs,
                num_channels, "Exporting test set (validation fold)",
                max_slices=max_slices,
                num_workers=num_workers
            )
        else:
            val_case_ids = export_dataset(
                val_dataset, imagesTs, labelsTs,
                num_channels, "Exporting test set (validation fold)",
                max_slices=max_slices
            )
    else:
        print("Skipping test set export (export_test=false)")
    
    # Free validation data references
    del val_dataset
    gc.collect()
    
    # === 9. Generate dataset.json ===
    channel_names = {
        str(i): cfg.dataset.modalities[i] 
        for i in range(num_channels)
    }
    
    # numTraining: use exported count if we exported, otherwise count from disk
    # This supports the resume scenario where only test is exported but
    # dataset.json still needs the correct numTraining from a previous run
    if export_train:
        num_training = len(train_case_ids)
    else:
        num_training = count_files_in_directory(labelsTr, "*.nii.gz")
        print(f"Counted {num_training} existing training labels on disk")
    
    num_test = len(val_case_ids)
    
    dataset_json = {
        "channel_names": channel_names,
        "labels": {
            "background": 0,
            "lesion": 1
        },
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "description": f"ISLES24 2D slices preprocessed with MedSegDiff pipeline",
        "reference": "Converted from MedSegDiff preprocessing",
        "source_config": {
            "dataset": cfg.dataset.name,
            "modalities": list(cfg.dataset.modalities),
            "image_size": cfg.model.image_size,
            "fold": cfg.dataset.fold,
            "test_mode": is_test_mode,
            "num_test": num_test
        }
    }
    
    with open(dataset_folder / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)
    
    # === 10. Summary ===
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"{'='*60}")
    print(f"Training slices: {num_training}")
    print(f"Test slices (fold {cfg.dataset.fold}): {num_test}")
    if is_test_mode:
        print(f"\n⚠️  TEST MODE: Only exported {max_slices} slices per split")
        print(f"   Run with nnunet.test=false for full export")
    print(f"\nOutput structure:")
    print(f"  {dataset_folder}/")
    print(f"  ├── dataset.json")
    print(f"  ├── imagesTr/  ({num_training * num_channels} files)")
    print(f"  ├── labelsTr/  ({num_training} files)")
    print(f"  ├── imagesTs/  ({num_test * num_channels} files)")
    print(f"  └── labelsTs/  ({num_test} files)")
    # Build helpful paths for next steps
    dataset_id = cfg.nnunet.dataset_id
    pred_dir = f"/mnt/outputs/nnunet_results/predictionsTs_{cfg.nnunet.dataset_name}"
    
    print(f"\nNext steps — Raw nnU-Net commands:")
    print(f"  1. export nnUNet_raw={output_base}")
    print(f"  2. nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity")
    print(f"  3. nnUNetv2_train {dataset_id} 2d all")
    print(f"  4. nnUNetv2_predict -i {imagesTs} -o {pred_dir} -d {dataset_id} -c 2d -f all")
    print(f"  5. python3 -m scripts.nnunet.compute_segmentation_metrics_for_nnunet_2d_predictions \\")
    print(f"       --pred-dir {pred_dir} --gt-dir {labelsTs}")
    
    print(f"\nNext steps — SLURM runners:")
    print(f"  1. python3 -m scripts.nnunet.slurm_runners.run_nnunet_command preprocess \\")
    print(f"       -d {dataset_id} --verify")
    print(f"  2. python3 -m scripts.nnunet.slurm_runners.run_nnunet_command train \\")
    print(f"       -d {dataset_id} -c 2d -f all")
    print(f"  3. python3 -m scripts.nnunet.slurm_runners.run_nnunet_command predict \\")
    print(f"       -d {dataset_id} -i {imagesTs} -o {pred_dir}")
    print(f"  4. python3 -m scripts.nnunet.slurm_runners.run_compute_segmentation_metrics_for_nnunet_2d_predictions \\")
    print(f"       --pred-dir {pred_dir} --gt-dir {labelsTs}")


if __name__ == "__main__":
    main()

