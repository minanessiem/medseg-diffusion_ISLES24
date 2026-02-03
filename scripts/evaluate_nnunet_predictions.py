#!/usr/bin/env python3
"""
Evaluate nnU-Net predictions against ground truth labels.

Computes standard 2D segmentation metrics (Dice, Precision, Recall, F1, F2)
by comparing NIfTI prediction files against ground truth labels exported
via our preprocessing pipeline.

Usage:
    python3 -m scripts.evaluate_nnunet_predictions \
        --pred-dir ../nnUNet_exports/predictionsTs/ \
        --gt-dir nnunet_raw/Dataset050_isles24/labelsTs/

Examples:
    # Evaluate predictions from a trained nnU-Net model
    python3 -m scripts.evaluate_nnunet_predictions \
        --pred-dir /path/to/nnUNet_results/predictionsTs \
        --gt-dir /path/to/nnunet_raw/Dataset050_isles24/labelsTs
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.metrics import (
    Dice2DForegroundOnly,
    VoxelPrecision2D,
    VoxelSensitivity2D,
    VoxelF1Score2D,
    VoxelF2Score2D,
)


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate nnU-Net predictions against ground truth labels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--pred-dir',
        type=Path,
        required=True,
        help='Directory containing nnU-Net prediction NIfTI files (*.nii.gz)'
    )
    parser.add_argument(
        '--gt-dir',
        type=Path,
        required=True,
        help='Directory containing ground truth label NIfTI files (*.nii.gz)'
    )
    
    return parser.parse_args()


# ============================================================================
# Data Loading
# ============================================================================

def load_nifti_pairs(
    pred_dir: Path,
    gt_dir: Path
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]:
    """
    Load paired predictions and ground truths from NIfTI files.
    
    Matches files by case ID (filename stem without .nii.gz extension).
    
    Args:
        pred_dir: Directory containing prediction files
        gt_dir: Directory containing ground truth files
        
    Returns:
        Tuple of (predictions, ground_truths, case_ids)
        - predictions: List of tensors with shape [1, H, W]
        - ground_truths: List of tensors with shape [1, H, W]
        - case_ids: List of case identifier strings
    """
    predictions = []
    ground_truths = []
    case_ids = []
    
    # Get all ground truth files
    gt_files = sorted(gt_dir.glob("*.nii.gz"))
    
    if not gt_files:
        raise FileNotFoundError(f"No .nii.gz files found in ground truth directory: {gt_dir}")
    
    # Build prediction file lookup
    pred_files = {f.name.replace(".nii.gz", ""): f for f in pred_dir.glob("*.nii.gz")}
    
    if not pred_files:
        raise FileNotFoundError(f"No .nii.gz files found in prediction directory: {pred_dir}")
    
    print(f"  Found {len(pred_files)} prediction files")
    print(f"  Found {len(gt_files)} ground truth files")
    
    missing_count = 0
    
    for gt_file in tqdm(gt_files, desc="  Loading data"):
        # Extract case ID from filename
        case_id = gt_file.name.replace(".nii.gz", "")
        
        # Check if prediction exists
        if case_id not in pred_files:
            missing_count += 1
            continue
        
        pred_file = pred_files[case_id]
        
        # Load NIfTI data
        gt_data = nib.load(gt_file).get_fdata()
        pred_data = nib.load(pred_file).get_fdata()
        
        # Convert to torch tensors with shape [1, H, W]
        # Input NIfTI shape is [H, W, 1], squeeze last dim and add channel dim
        gt_tensor = torch.from_numpy(gt_data).float().squeeze(-1).unsqueeze(0)
        pred_tensor = torch.from_numpy(pred_data).float().squeeze(-1).unsqueeze(0)
        
        # Validate shapes match
        if gt_tensor.shape != pred_tensor.shape:
            print(f"  Warning: Shape mismatch for {case_id}: "
                  f"pred={pred_tensor.shape}, gt={gt_tensor.shape}")
            continue
        
        predictions.append(pred_tensor)
        ground_truths.append(gt_tensor)
        case_ids.append(case_id)
    
    if missing_count > 0:
        print(f"  Warning: {missing_count} ground truth files had no matching prediction")
    
    print(f"  Matched {len(case_ids)} pairs")
    
    return predictions, ground_truths, case_ids


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_metrics(
    predictions: List[torch.Tensor],
    ground_truths: List[torch.Tensor]
) -> Dict:
    """
    Compute all metrics over prediction/ground truth pairs.
    
    Tracks foreground vs empty slices separately. Dice is computed only
    over slices with foreground (matching training behavior).
    
    Args:
        predictions: List of prediction tensors [1, H, W]
        ground_truths: List of ground truth tensors [1, H, W]
        
    Returns:
        Dictionary with metric statistics and slice counts
    """
    # Initialize metric instances
    dice_metric = Dice2DForegroundOnly()
    precision_metric = VoxelPrecision2D()
    recall_metric = VoxelSensitivity2D()
    f1_metric = VoxelF1Score2D()
    f2_metric = VoxelF2Score2D()
    
    # Track per-slice values for std computation
    dice_values = []
    precision_values = []
    recall_values = []
    f1_values = []
    f2_values = []
    
    foreground_count = 0
    empty_count = 0
    
    for pred, gt in tqdm(zip(predictions, ground_truths), 
                         total=len(predictions),
                         desc="  Computing metrics"):
        # Check if slice has foreground
        has_foreground = bool((gt > 0.5).sum() > 0)
        
        if has_foreground:
            foreground_count += 1
            # Dice only computed for foreground slices
            dice_val = dice_metric(pred, gt)
            dice_values.append(dice_val.item())
        else:
            empty_count += 1
        
        # Other metrics computed for all slices
        precision_values.append(precision_metric(pred, gt).item())
        recall_values.append(recall_metric(pred, gt).item())
        f1_values.append(f1_metric(pred, gt).item())
        f2_values.append(f2_metric(pred, gt).item())
    
    # Aggregate results
    results = {
        "slice_counts": {
            "total": len(predictions),
            "foreground": foreground_count,
            "empty": empty_count,
        },
        "metrics": {
            "dice_fg": {
                "mean": float(np.mean(dice_values)) if dice_values else 0.0,
                "std": float(np.std(dice_values)) if dice_values else 0.0,
            },
            "precision": {
                "mean": float(np.mean(precision_values)),
                "std": float(np.std(precision_values)),
            },
            "recall": {
                "mean": float(np.mean(recall_values)),
                "std": float(np.std(recall_values)),
            },
            "f1": {
                "mean": float(np.mean(f1_values)),
                "std": float(np.std(f1_values)),
            },
            "f2": {
                "mean": float(np.mean(f2_values)),
                "std": float(np.std(f2_values)),
            },
        }
    }
    
    return results


# ============================================================================
# Output Generation
# ============================================================================

def save_results(
    results: Dict,
    pred_dir: Path,
    gt_dir: Path
):
    """
    Save evaluation results to JSON and text files.
    
    Files are saved to the prediction directory.
    
    Args:
        results: Dictionary with computed metrics
        pred_dir: Prediction directory (output location)
        gt_dir: Ground truth directory (for metadata)
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Build full results with metadata
    full_results = {
        "metadata": {
            "pred_dir": str(pred_dir.resolve()),
            "gt_dir": str(gt_dir.resolve()),
            "timestamp": timestamp,
            "num_pairs": results["slice_counts"]["total"],
        },
        **results
    }
    
    # Save JSON
    json_path = pred_dir / "evaluation_results.json"
    with open(json_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    # Save human-readable summary
    txt_path = pred_dir / "evaluation_summary.txt"
    
    sc = results["slice_counts"]
    m = results["metrics"]
    fg_pct = 100 * sc["foreground"] / sc["total"] if sc["total"] > 0 else 0
    
    lines = [
        "nnU-Net Prediction Evaluation Results",
        "=" * 50,
        f"Timestamp: {timestamp}",
        f"Predictions: {pred_dir.resolve()}",
        f"Ground Truth: {gt_dir.resolve()}",
        "",
        "Slice Counts:",
        f"  Total:      {sc['total']}",
        f"  Foreground: {sc['foreground']} ({fg_pct:.1f}%)",
        f"  Empty:      {sc['empty']} ({100-fg_pct:.1f}%)",
        "",
        "Metrics (micro-averaged):",
        f"  Dice (foreground only): {m['dice_fg']['mean']:.4f} (±{m['dice_fg']['std']:.4f})",
        f"  Precision:              {m['precision']['mean']:.4f} (±{m['precision']['std']:.4f})",
        f"  Recall/Sensitivity:     {m['recall']['mean']:.4f} (±{m['recall']['std']:.4f})",
        f"  F1 Score:               {m['f1']['mean']:.4f} (±{m['f1']['std']:.4f})",
        f"  F2 Score:               {m['f2']['mean']:.4f} (±{m['f2']['std']:.4f})",
        "",
        "=" * 50,
    ]
    
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"\nResults saved to:")
    print(f"  {json_path}")
    print(f"  {txt_path}")


def print_summary(results: Dict):
    """Print evaluation summary to console."""
    sc = results["slice_counts"]
    m = results["metrics"]
    fg_pct = 100 * sc["foreground"] / sc["total"] if sc["total"] > 0 else 0
    
    print("\n[3/3] Results")
    print("-" * 60)
    print(f"Slices evaluated: {sc['total']}")
    print(f"  Foreground slices: {sc['foreground']} ({fg_pct:.1f}%)")
    print(f"  Empty slices: {sc['empty']} ({100-fg_pct:.1f}%)")
    print()
    print("Metrics (micro-averaged):")
    print(f"  Dice (foreground only): {m['dice_fg']['mean']:.4f} (±{m['dice_fg']['std']:.4f})")
    print(f"  Precision:              {m['precision']['mean']:.4f} (±{m['precision']['std']:.4f})")
    print(f"  Recall/Sensitivity:     {m['recall']['mean']:.4f} (±{m['recall']['std']:.4f})")
    print(f"  F1 Score:               {m['f1']['mean']:.4f} (±{m['f1']['std']:.4f})")
    print(f"  F2 Score:               {m['f2']['mean']:.4f} (±{m['f2']['std']:.4f})")
    print("-" * 60)


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_arguments()
    
    # Validate directories exist
    if not args.pred_dir.exists():
        print(f"Error: Prediction directory does not exist: {args.pred_dir}")
        sys.exit(1)
    if not args.gt_dir.exists():
        print(f"Error: Ground truth directory does not exist: {args.gt_dir}")
        sys.exit(1)
    
    print("=" * 60)
    print("nnU-Net Prediction Evaluation")
    print("=" * 60)
    print(f"Predictions: {args.pred_dir}")
    print(f"Ground Truth: {args.gt_dir}")
    print("=" * 60)
    
    # 1. Load data
    print("\n[1/3] Loading data...")
    predictions, ground_truths, case_ids = load_nifti_pairs(args.pred_dir, args.gt_dir)
    
    if not predictions:
        print("Error: No valid prediction/ground truth pairs found")
        sys.exit(1)
    
    # 2. Compute metrics
    print("\n[2/3] Computing metrics...")
    results = compute_metrics(predictions, ground_truths)
    
    # 3. Output results
    print_summary(results)
    save_results(results, args.pred_dir, args.gt_dir)
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

