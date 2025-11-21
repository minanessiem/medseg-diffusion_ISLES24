"""
Augmentation visualization test.

This test creates side-by-side visualizations of original vs augmented images
to verify that augmentations produce realistic outputs. Not a traditional unit test -
serves as a visual debugging and validation tool.

Usage:
    python -m tests.test_augmentation_visualization
    
Output:
    Saves images to tests/augmentation_visualizations/{aug_config}/
"""

import torch
import os
import sys
import shutil
import hydra
from omegaconf import OmegaConf
from src.data.loaders import get_dataloaders
from src.utils.logger import Logger
from torch.utils.tensorboard import SummaryWriter

def run_visualization(aug_config_name):
    """
    Visualize augmentation effects for a given config name.
    
    Args:
        aug_config_name: Name of augmentation config (none, light_2d, aggressive_2d)
    """
    print(f"\n{'='*60}")
    print(f"Visualizing augmentation: {aug_config_name}")
    print(f"{'='*60}\n")
    
    # Initialize Hydra and compose config
    # We base this on 'local' config which should have valid paths for the environment
    with hydra.initialize(config_path="../configs", version_base=None):
        # Override augmentation and ensure minimal resources for testing
        overrides = [
            f"augmentation={aug_config_name}",
            "dataset.num_train_workers=0",
            "dataset.num_valid_workers=0",
            "dataset.num_test_workers=0",
            "dataset.train_batch_size=1",
            "dataset.test_batch_size=1",
            "validation.val_batch_size=1",
            "training.max_steps=1"  # Minimal steps just to load config
        ]
        
        try:
            cfg = hydra.compose(config_name="local", overrides=overrides)
            # Allow adding new keys (needed for get_dataloaders aliasing)
            OmegaConf.set_struct(cfg, False)
        except Exception as e:
            print(f"Failed to compose config: {e}")
            return

    # Get dataloader with augmentation
    # Use environment path for printing since dataset.dir isn't aliased yet
    data_path = cfg.environment.dataset.dir if 'environment' in cfg else 'UNKNOWN'
    print(f"Loading dataset from {data_path}...")
    try:
        dataloaders = get_dataloaders(cfg)
        train_dl = dataloaders['train']
    except Exception as e:
        print(f"Skipping visualization for {aug_config_name}: Failed to load dataset. Error: {e}")
        return
    
    # Create output directory
    output_dir = f"tests/augmentation_visualizations/{aug_config_name}/"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create logger with tensorboard writer
    writer = SummaryWriter(log_dir=output_dir)
    
    # Create minimal logging config with required keys
    logging_cfg = OmegaConf.create({
        'enable_image_logging': True,
        'enable_image_labels': True,
        'label_color': [255, 0, 0],
        'label_font_size': 22,
        'label_position': 'lower_right',
        'label_offset': [10, 10]
    })
    
    logger = Logger(
        log_dir=output_dir,
        enabled_outputs=['tensorboard'],
        log_interval=1,
        table_format='stdout',
        writer=writer,
        cfg=logging_cfg
    )
    
    # Collect N non-empty samples
    num_samples = 5
    collected_images = []
    collected_labels = []
    
    print(f"Collecting {num_samples} samples with non-empty masks...")
    try:
        for idx, (img, label, case_id) in enumerate(train_dl):
            if len(collected_images) >= num_samples:
                break
            
            # Skip empty masks
            if label.sum() == 0:
                continue
            
            collected_images.append(img[0])  # Remove batch dim
            collected_labels.append(label[0])
            # Handle tuple case_id if batch size > 1, though here it is 1
            cid = case_id[0] if isinstance(case_id, (list, tuple)) else case_id
            print(f"  Sample {len(collected_images)}: {cid}, mask sum: {label.sum().item():.0f}")
    except Exception as e:
        print(f"Error during dataloading: {e}")
        return
    
    if len(collected_images) == 0:
        print("WARNING: No non-empty samples found.")
        return
    
    print(f"\nCollected {len(collected_images)} samples")
    
    # For each sample, create side-by-side visualization
    all_images = []
    labels = []
    num_modalities = collected_images[0].shape[0]
    
    print("Generating augmented versions...")
    for i in range(len(collected_images)):
        # Original modality channels
        for c in range(num_modalities):
            all_images.append(collected_images[i][c:c+1])
            labels.append(f"Orig_Mod{c}")
        
        # Original mask
        all_images.append(collected_labels[i])
        labels.append("Orig_Mask")
        
        # Augmented versions
        if train_dl.dataset.augmentation is not None:
            # Apply augmentation to get modified version
            data_dict = {
                'image': collected_images[i],
                'label': collected_labels[i]
            }
            # Note: We don't manually seed here to show variation, 
            # or we could if specific determinism was needed for the viz
            aug_dict = train_dl.dataset.augmentation(data_dict)
            aug_img = aug_dict['image']
            aug_label = aug_dict['label']
            
            # Augmented channels
            for c in range(num_modalities):
                all_images.append(aug_img[c:c+1])
                labels.append(f"Aug_Mod{c}")
            
            # Augmented mask
            all_images.append(aug_label)
            labels.append("Aug_Mask")
        else:
            # If no augmentation, duplicate originals to keep grid aligned
            for c in range(num_modalities):
                all_images.append(collected_images[i][c:c+1])
                labels.append(f"NoAug_Mod{c}")
            all_images.append(collected_labels[i])
            labels.append("NoAug_Mask")
    
    # Log grid
    # Columns per sample: (num_modalities + 1) for original + (num_modalities + 1) for augmented
    per_sample_ncol = (num_modalities + 1) * 2 
    
    print(f"Logging visualization grid...")
    logger.log_image_grid(
        f"Augmentation_Visualization_{aug_config_name}",
        all_images,
        0,  # global_step (positional)
        metrics=None,
        grid_layout='horizontal',
        labels=labels,
        per_sample_ncol=per_sample_ncol
    )
    
    logger.close()
    writer.close()
    
    print(f"\n✓ Visualization saved to {output_dir}")
    print(f"  View with: tensorboard --logdir {output_dir}\n")

if __name__ == "__main__":
    presets = ["none", "light_2d", "aggressive_2d"]
    for preset in presets:
        run_visualization(preset)
