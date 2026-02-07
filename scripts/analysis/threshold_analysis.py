#!/usr/bin/env python3
"""
Threshold analysis for trained segmentation models.

Evaluates a trained discriminative model across multiple thresholds,
computing metrics, generating plots, and creating comparison visualizations.

Usage:
    python scripts/analysis/threshold_analysis.py \
        --run-dir <path> \
        --model-name <name> \
        [options]

Examples:
    # Basic usage with default thresholds (0.05 to 0.95, step 0.05)
    python scripts/analysis/threshold_analysis.py \
        --run-dir outputs/my_run_2026-01-17/ \
        --model-name best_model_step_002000_dice_2d_fg_0.1815

    # Custom threshold range
    python scripts/analysis/threshold_analysis.py \
        --run-dir outputs/my_run/ \
        --model-name best_model \
        --thresholds 0.1:0.9:0.1

    # Comma-separated specific thresholds
    python scripts/analysis/threshold_analysis.py \
        --run-dir outputs/my_run/ \
        --model-name best_model \
        --thresholds 0.3,0.4,0.5,0.6,0.7
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.analysis.metrics_registry import compute_metrics_at_threshold, get_all_metrics
from src.utils.ensemble import mean_ensemble, soft_staple


# ============================================================================
# Argument Parsing
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Threshold analysis for trained segmentation models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        '--run-dir', 
        required=True,
        help='Path to run directory containing checkpoints and .hydra config'
    )
    parser.add_argument(
        '--model-name', 
        required=True,
        help='Model checkpoint name (without .pth extension)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--thresholds',
        type=str,
        default='0.05:0.95:0.05',
        help='Threshold range as "start:stop:step" or comma-separated values (default: 0.05:0.95:0.05)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: {run_dir}/analysis/threshold_sweep/{model_name}_{timestamp})'
    )
    parser.add_argument(
        '--save-per-sample',
        action='store_true',
        help='Save per-sample metrics CSV (can be large)'
    )
    parser.add_argument(
        '--num-visualizations',
        type=int,
        default=4,
        help='Number of comparison visualization images to generate (default: 4)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (default: cuda if available, else cpu)'
    )
    
    # Ensemble options (for diffusion models)
    parser.add_argument(
        '--ensemble-samples',
        type=int,
        default=1,
        help='Number of samples to ensemble for diffusion models (default: 1, no ensemble)'
    )
    parser.add_argument(
        '--ensemble-method',
        type=str,
        default='soft_staple',
        choices=['mean', 'soft_staple'],
        help="Ensemble method: 'mean' or 'soft_staple' (default: soft_staple)"
    )
    parser.add_argument(
        '--staple-max-iters',
        type=int,
        default=5,
        help='Max iterations for soft_staple algorithm (default: 5)'
    )
    parser.add_argument(
        '--staple-tolerance',
        type=float,
        default=0.02,
        help='Convergence tolerance for soft_staple (default: 0.02)'
    )
    
    return parser.parse_args()


# ============================================================================
# Config Loading
# ============================================================================

def load_config_from_run_dir(run_dir: str) -> DictConfig:
    """
    Load Hydra configuration from a run directory.
    
    Args:
        run_dir: Path to the run directory containing .hydra/config.yaml
        
    Returns:
        Loaded and resolved configuration
        
    Raises:
        FileNotFoundError: If config.yaml not found
    """
    hydra_dir = os.path.join(run_dir, '.hydra')
    config_path = os.path.join(hydra_dir, 'config.yaml')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config not found: {config_path}\n"
            f"Make sure the run directory contains a .hydra subdirectory with config.yaml"
        )
    
    cfg = OmegaConf.load(config_path)
    return cfg


# ============================================================================
# Model Loading
# ============================================================================

def find_checkpoint(run_dir: str, model_name: str) -> str:
    """
    Search for model checkpoint in standard locations.
    
    Args:
        run_dir: Path to run directory
        model_name: Model checkpoint name (without .pth)
        
    Returns:
        Full path to checkpoint file
        
    Raises:
        FileNotFoundError: If checkpoint not found in any location
    """
    # Add .pth extension if not present
    if not model_name.endswith('.pth'):
        model_name = f'{model_name}.pth'
    
    # Search locations in priority order
    candidates = [
        os.path.join(run_dir, 'models', 'best', model_name),
        os.path.join(run_dir, 'models', 'checkpoints', model_name),
        os.path.join(run_dir, 'models', model_name),
    ]
    
    for path in candidates:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError(
        f"Checkpoint not found: {model_name}\n"
        f"Searched locations:\n" + 
        "\n".join(f"  - {p}" for p in candidates)
    )


def is_discriminative_model(cfg: DictConfig) -> bool:
    """Check if the model is discriminative (single forward pass) vs diffusion (sampling loop)."""
    diffusion_type = cfg.get('diffusion', {}).get('type', 'Discriminative')
    return diffusion_type == 'Discriminative'


def load_model(cfg: DictConfig, checkpoint_path: str, device: str) -> nn.Module:
    """
    Build model/diffusion from config and load checkpoint weights.
    
    For discriminative models: Returns the model directly
    For diffusion models: Returns the full diffusion wrapper with sample() method
    
    Args:
        cfg: Hydra configuration
        checkpoint_path: Path to model checkpoint
        device: Device to load model onto
        
    Returns:
        Model or Diffusion wrapper in evaluation mode
    """
    from src.models.model_factory import build_model
    from src.diffusion.diffusion import Diffusion
    
    # Build model architecture
    unet = build_model(cfg)
    
    # For diffusion models, we need the full diffusion wrapper
    if not is_discriminative_model(cfg):
        print(f"  Building diffusion wrapper (type: {cfg.diffusion.type})...")
        diffusion = Diffusion.build_diffusion(unet, cfg, device)
        model = diffusion
    else:
        model = unet
    
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Handle potential wrapper keys (e.g., from DDP or EMA)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    # Get the expected keys from the model to determine correct prefix handling
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(state_dict.keys())
    
    # Try to find the correct prefix mapping
    def strip_prefix(state_dict: dict, prefix: str) -> dict:
        """Remove a prefix from all keys in state_dict."""
        return {k[len(prefix):] if k.startswith(prefix) else k: v 
                for k, v in state_dict.items()}
    
    # Common prefixes to try stripping (in order of priority)
    prefixes_to_try = [
        'module.',                    # DataParallel/DDP
        'wrapped_model.base_model.',  # EMA wrapper
        'model.model.',               # Double adapter wrapping (SwinUNETR)
        'model.',                     # Single model wrapper (only if not diffusion)
    ]
    
    # Try each prefix and see if it results in matching keys
    best_match = state_dict
    best_overlap = len(model_keys & checkpoint_keys)
    
    for prefix in prefixes_to_try:
        if any(k.startswith(prefix) for k in state_dict.keys()):
            stripped = strip_prefix(state_dict, prefix)
            overlap = len(model_keys & set(stripped.keys()))
            if overlap > best_overlap:
                best_match = stripped
                best_overlap = overlap
                print(f"  Stripped prefix '{prefix}' -> {overlap} matching keys")
    
    state_dict = best_match
    
    # Final attempt: if still no match, try strict=False loading with diagnostic
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"  Warning: Strict loading failed, trying non-strict...")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    model.to(device)
    model.eval()
    
    return model


# ============================================================================
# Inference
# ============================================================================

def run_inference(
    model: nn.Module, 
    cfg: DictConfig, 
    device: str,
    args: argparse.Namespace
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Run model inference on validation set.
    
    Automatically detects model type:
    - Discriminative models: Direct forward pass
    - Diffusion models: Uses sample() method with full sampling loop
    
    Supports ensemble inference for diffusion models when args.ensemble_samples >= 2.
    
    Args:
        model: Trained model/diffusion in evaluation mode
        cfg: Configuration with dataset settings
        device: Device to run inference on
        args: Command-line arguments with ensemble settings
        
    Returns:
        Tuple of (predictions, ground_truths, modalities) as lists of tensors
    """
    from src.data.loaders import get_dataloaders
    
    dataloaders = get_dataloaders(cfg)
    val_loader = dataloaders['val']
    
    # Determine inference method based on model type
    use_diffusion_sampling = not is_discriminative_model(cfg)
    use_ensemble = use_diffusion_sampling and args.ensemble_samples >= 2
    
    if use_diffusion_sampling:
        if use_ensemble:
            print(f"  Using diffusion sampling with ensemble ({args.ensemble_samples} samples, method={args.ensemble_method})...")
        else:
            print("  Using diffusion sampling loop for inference...")
    
    all_predictions = []
    all_ground_truths = []
    all_modalities = []
    
    # Progress bar description
    desc = "Running inference"
    if use_ensemble:
        desc = f"Running inference ({args.ensemble_samples} samples/input)"
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=desc):
            # Unpack batch - format is (image, mask, path)
            img = batch[0].to(device)
            mask = batch[1]
            
            # Forward pass - different for discriminative vs diffusion
            if use_diffusion_sampling:
                if use_ensemble:
                    # Ensemble sampling: generate N samples and combine
                    samples = []
                    for _ in range(args.ensemble_samples):
                        sample = model.sample(img, disable_tqdm=True)
                        samples.append(sample)
                    samples = torch.stack(samples, dim=0)  # [N, B, C, H, W]
                    
                    if args.ensemble_method == 'mean':
                        pred = mean_ensemble(samples)
                    else:  # soft_staple
                        pred = soft_staple(
                            samples,
                            max_iters=args.staple_max_iters,
                            tolerance=args.staple_tolerance
                        )
                else:
                    # Single sample (default behavior)
                    pred = model.sample(img, disable_tqdm=True)
            else:
                # Discriminative models: direct forward pass
                pred = model(img)
            
            # Handle output normalization based on model type
            if use_diffusion_sampling:
                # Diffusion models output values in ~[0,1] range directly
                # Small overflows are possible; clamp instead of sigmoid
                pred = pred.clamp(0, 1)
            else:
                # Discriminative models may output logits; apply sigmoid if needed
                if pred.min() < 0 or pred.max() > 1:
                    pred = torch.sigmoid(pred)
            
            # Store per-sample (unbatch)
            for i in range(pred.shape[0]):
                all_predictions.append(pred[i].cpu())
                all_ground_truths.append(mask[i].cpu())
                all_modalities.append(img[i].cpu())
    
    return all_predictions, all_ground_truths, all_modalities


# ============================================================================
# Threshold Sweep
# ============================================================================

def parse_thresholds(threshold_str: str) -> List[float]:
    """
    Parse threshold specification string.
    
    Args:
        threshold_str: Either "start:stop:step" or comma-separated values
        
    Returns:
        List of threshold values
    """
    if ':' in threshold_str:
        parts = threshold_str.split(':')
        if len(parts) != 3:
            raise ValueError(f"Invalid threshold format: {threshold_str}. Expected start:stop:step")
        start, stop, step = float(parts[0]), float(parts[1]), float(parts[2])
        # Use np.arange and round to avoid floating point issues
        thresholds = np.arange(start, stop + step/2, step)
        return [round(t, 4) for t in thresholds]
    else:
        return [float(t.strip()) for t in threshold_str.split(',')]


def compute_threshold_sweep(
    predictions: List[torch.Tensor],
    ground_truths: List[torch.Tensor],
    thresholds: List[float]
) -> Dict:
    """
    Compute all metrics at all thresholds.
    
    For foreground-only metrics (dice), only includes slices with ground truth
    foreground in the average, matching training validation behavior.
    
    Args:
        predictions: List of prediction tensors
        ground_truths: List of ground truth tensors
        thresholds: List of threshold values
        
    Returns:
        Dictionary with results for each threshold
    """
    results = {t: {'samples': [], 'has_foreground': []} for t in thresholds}
    
    for pred, gt in tqdm(zip(predictions, ground_truths), 
                         total=len(predictions), 
                         desc="Computing metrics"):
        # Track if this slice has foreground (for foreground-only metrics)
        has_fg = bool((gt > 0.5).sum() > 0)
        
        for t in thresholds:
            metrics = compute_metrics_at_threshold(pred, gt, t)
            results[t]['samples'].append(metrics)
            results[t]['has_foreground'].append(has_fg)
    
    # Aggregate statistics
    # Foreground-only metrics should only average over slices with foreground
    foreground_only_metrics = {'dice'}  # Add others if needed
    
    for t in thresholds:
        samples = results[t]['samples']
        has_fg_list = results[t]['has_foreground']
        
        if samples:
            metric_names = samples[0].keys()
            for metric_name in metric_names:
                values = [s[metric_name] for s in samples]
                
                if metric_name in foreground_only_metrics:
                    # Only average over slices with foreground (matching training)
                    fg_values = [v for v, has_fg in zip(values, has_fg_list) if has_fg]
                    if fg_values:
                        results[t][f'{metric_name}_mean'] = float(np.mean(fg_values))
                        results[t][f'{metric_name}_std'] = float(np.std(fg_values))
                    else:
                        results[t][f'{metric_name}_mean'] = 0.0
                        results[t][f'{metric_name}_std'] = 0.0
                else:
                    # All slices for other metrics
                    results[t][f'{metric_name}_mean'] = float(np.mean(values))
                    results[t][f'{metric_name}_std'] = float(np.std(values))
        
        # Count foreground slices for reference
        results[t]['foreground_slices'] = sum(has_fg_list)
        results[t]['total_slices'] = len(has_fg_list)
    
    return results


def find_optimal_thresholds(
    results: Dict, 
    thresholds: List[float]
) -> Dict:
    """
    Find threshold that maximizes each metric.
    
    Args:
        results: Results dictionary from compute_threshold_sweep
        thresholds: List of evaluated thresholds
        
    Returns:
        Dictionary with optimal threshold info for each metric
    """
    metric_names = ['dice', 'precision', 'recall', 'specificity', 'f1', 'f2']
    optimal = {}
    
    for metric in metric_names:
        mean_key = f'{metric}_mean'
        std_key = f'{metric}_std'
        
        # Find threshold with maximum mean value
        best_t = max(thresholds, key=lambda t: results[t].get(mean_key, 0))
        
        optimal[metric] = {
            'threshold': best_t,
            'value': results[best_t].get(mean_key, 0),
            'std': results[best_t].get(std_key, 0),
        }
    
    return optimal


# ============================================================================
# AUC Computation
# ============================================================================

def compute_auc_metrics(
    predictions: List[torch.Tensor],
    ground_truths: List[torch.Tensor]
) -> Tuple[float, float]:
    """
    Compute AUC-ROC and AUC-PR.
    
    Args:
        predictions: List of prediction tensors (sigmoid probabilities)
        ground_truths: List of ground truth tensors
        
    Returns:
        Tuple of (AUC-ROC, AUC-PR)
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    # Flatten all predictions and ground truths
    all_pred = torch.cat([p.flatten() for p in predictions]).numpy()
    all_gt = torch.cat([g.flatten() for g in ground_truths]).numpy()
    
    # Binarize ground truth
    all_gt_binary = (all_gt > 0.5).astype(int)
    
    # Check if we have both classes
    if len(np.unique(all_gt_binary)) < 2:
        print("  Warning: Only one class present in ground truth, AUC undefined")
        return 0.5, 0.0
    
    auc_roc = roc_auc_score(all_gt_binary, all_pred)
    auc_pr = average_precision_score(all_gt_binary, all_pred)
    
    return float(auc_roc), float(auc_pr)


# ============================================================================
# Output Generation
# ============================================================================

def save_csv_results(
    results: Dict,
    thresholds: List[float],
    output_dir: str,
    save_per_sample: bool = False
):
    """Save metrics to CSV files."""
    import csv
    
    # Aggregated metrics per threshold
    csv_path = os.path.join(output_dir, 'metrics_per_threshold.csv')
    
    metric_names = ['dice', 'precision', 'recall', 'specificity', 'f1', 'f2']
    header = ['threshold']
    for m in metric_names:
        header.extend([f'{m}_mean', f'{m}_std'])
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for t in thresholds:
            row = [t]
            for m in metric_names:
                row.append(results[t].get(f'{m}_mean', 0))
                row.append(results[t].get(f'{m}_std', 0))
            writer.writerow(row)
    
    print(f"  Saved: {csv_path}")
    
    # Optional per-sample metrics
    if save_per_sample:
        per_sample_path = os.path.join(output_dir, 'metrics_per_sample.csv')
        
        with open(per_sample_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['sample_idx', 'threshold'] + metric_names
            writer.writerow(header)
            
            # Data
            for t in thresholds:
                for idx, sample_metrics in enumerate(results[t]['samples']):
                    row = [idx, t] + [sample_metrics[m] for m in metric_names]
                    writer.writerow(row)
        
        print(f"  Saved: {per_sample_path}")


def save_json_summary(
    optimal: Dict,
    auc_roc: float,
    auc_pr: float,
    results: Dict,
    args: argparse.Namespace,
    thresholds: List[float],
    num_samples: int,
    output_dir: str
):
    """Save optimal thresholds and summary to JSON."""
    # Get default threshold (0.5) metrics for comparison
    default_threshold = 0.5
    closest_default = min(thresholds, key=lambda t: abs(t - default_threshold))
    
    dice_at_default = results[closest_default].get('dice_mean', 0)
    dice_at_optimal = optimal['dice']['value']
    improvement = dice_at_optimal - dice_at_default
    improvement_pct = (improvement / dice_at_default * 100) if dice_at_default > 0 else 0
    
    # Get foreground slice count from results
    sample_threshold = thresholds[0]
    fg_slices = results[sample_threshold].get('foreground_slices', num_samples)
    
    summary = {
        'model_name': args.model_name,
        'run_dir': args.run_dir,
        'analysis_timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        'num_samples': num_samples,
        'foreground_slices': fg_slices,
        'thresholds_evaluated': thresholds,
        'optimal': optimal,
        'auc': {
            'roc': auc_roc,
            'pr': auc_pr,
        },
        'comparison_vs_default': {
            'default_threshold': closest_default,
            'dice_at_default': dice_at_default,
            'dice_at_optimal': dice_at_optimal,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
        }
    }
    
    # Add ensemble configuration if used
    if args.ensemble_samples >= 2:
        summary['ensemble'] = {
            'enabled': True,
            'num_samples': args.ensemble_samples,
            'method': args.ensemble_method,
        }
        if args.ensemble_method == 'soft_staple':
            summary['ensemble']['soft_staple'] = {
                'max_iters': args.staple_max_iters,
                'tolerance': args.staple_tolerance,
            }
    else:
        summary['ensemble'] = {
            'enabled': False,
            'num_samples': 1,
        }
    
    json_path = os.path.join(output_dir, 'optimal_thresholds.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Saved: {json_path}")


def save_summary_text(
    optimal: Dict,
    auc_roc: float,
    auc_pr: float,
    results: Dict,
    args: argparse.Namespace,
    thresholds: List[float],
    num_samples: int,
    output_dir: str
):
    """Save human-readable summary text file."""
    # Get default threshold metrics
    default_threshold = 0.5
    closest_default = min(thresholds, key=lambda t: abs(t - default_threshold))
    dice_at_default = results[closest_default].get('dice_mean', 0)
    dice_at_optimal = optimal['dice']['value']
    improvement_pct = ((dice_at_optimal - dice_at_default) / dice_at_default * 100) if dice_at_default > 0 else 0
    
    lines = [
        "Threshold Analysis Report",
        "=" * 50,
        f"Model: {args.model_name}",
        f"Run: {args.run_dir}",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Validation Set: {num_samples} slices",
    ]
    
    # Add foreground slice info if available
    sample_threshold = thresholds[0]
    if 'foreground_slices' in results[sample_threshold]:
        fg_slices = results[sample_threshold]['foreground_slices']
        lines.append(f"  Foreground slices: {fg_slices} ({100*fg_slices/num_samples:.1f}%)")
    
    # Add ensemble info if used
    if args.ensemble_samples >= 2:
        lines.extend([
            "",
            "Ensemble Configuration:",
            f"  Samples: {args.ensemble_samples}",
            f"  Method: {args.ensemble_method}",
        ])
        if args.ensemble_method == 'soft_staple':
            lines.append(f"  STAPLE: max_iters={args.staple_max_iters}, tolerance={args.staple_tolerance}")
    
    lines.extend([
        "",
        "Optimal Thresholds:",
    ])
    
    for metric in ['dice', 'precision', 'recall', 'specificity', 'f1', 'f2']:
        t = optimal[metric]['threshold']
        v = optimal[metric]['value']
        s = optimal[metric]['std']
        lines.append(f"  {metric.capitalize():12} τ={t:.2f} → {v:.4f} (±{s:.4f})")
    
    lines.extend([
        "",
        "AUC Metrics:",
        f"  AUC-ROC: {auc_roc:.4f}",
        f"  AUC-PR:  {auc_pr:.4f}",
        "",
        f"Comparison (Default τ={closest_default:.2f} vs Optimal τ={optimal['dice']['threshold']:.2f}):",
        f"  Dice: {dice_at_default:.4f} → {dice_at_optimal:.4f} ({improvement_pct:+.1f}%)",
    ])
    
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"  Saved: {summary_path}")


# ============================================================================
# Plot Generation
# ============================================================================

def generate_plots(
    results: Dict,
    thresholds: List[float],
    optimal: Dict,
    auc_roc: float,
    auc_pr: float,
    predictions: List[torch.Tensor],
    ground_truths: List[torch.Tensor],
    output_dir: str
):
    """Generate all analysis plots."""
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 1: Threshold vs Metrics
    plot_threshold_vs_metrics(results, thresholds, optimal, plots_dir)
    
    # Plot 2: ROC Curve
    plot_roc_curve(predictions, ground_truths, auc_roc, plots_dir)
    
    # Plot 3: PR Curve
    plot_pr_curve(predictions, ground_truths, auc_pr, plots_dir)
    
    print(f"  Saved plots to: {plots_dir}")


def plot_threshold_vs_metrics(
    results: Dict,
    thresholds: List[float],
    optimal: Dict,
    plots_dir: str
):
    """Plot all metrics vs threshold."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metric_configs = [
        ('dice', 'Dice', '#2ecc71', '-'),
        ('precision', 'Precision', '#3498db', '--'),
        ('recall', 'Recall', '#e74c3c', '--'),
        ('specificity', 'Specificity', '#9b59b6', ':'),
        ('f1', 'F1', '#f39c12', '-'),
        ('f2', 'F2', '#1abc9c', '-.'),
    ]
    
    for metric, label, color, linestyle in metric_configs:
        values = [results[t].get(f'{metric}_mean', 0) for t in thresholds]
        ax.plot(thresholds, values, label=label, color=color, linestyle=linestyle, linewidth=2)
    
    # Mark optimal Dice threshold
    opt_dice_t = optimal['dice']['threshold']
    ax.axvline(x=opt_dice_t, color='gray', linestyle='--', alpha=0.7, 
               label=f'Optimal Dice (τ={opt_dice_t:.2f})')
    
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Segmentation Metrics vs Threshold', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    # Save in both formats
    for ext in ['png', 'pdf']:
        path = os.path.join(plots_dir, f'threshold_vs_metrics.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
    
    plt.close(fig)


def plot_roc_curve(
    predictions: List[torch.Tensor],
    ground_truths: List[torch.Tensor],
    auc_roc: float,
    plots_dir: str
):
    """Plot ROC curve with AUC annotation."""
    from sklearn.metrics import roc_curve
    
    # Flatten all data
    all_pred = torch.cat([p.flatten() for p in predictions]).numpy()
    all_gt = torch.cat([g.flatten() for g in ground_truths]).numpy()
    all_gt_binary = (all_gt > 0.5).astype(int)
    
    # Compute ROC curve
    fpr, tpr, _ = roc_curve(all_gt_binary, all_pred)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(fpr, tpr, color='#3498db', linewidth=2, 
            label=f'ROC Curve (AUC = {auc_roc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.7, 
            label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    
    plt.tight_layout()
    
    for ext in ['png', 'pdf']:
        path = os.path.join(plots_dir, f'roc_curve.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
    
    plt.close(fig)


def plot_pr_curve(
    predictions: List[torch.Tensor],
    ground_truths: List[torch.Tensor],
    auc_pr: float,
    plots_dir: str
):
    """Plot Precision-Recall curve with AUC-PR annotation."""
    from sklearn.metrics import precision_recall_curve
    
    # Flatten all data
    all_pred = torch.cat([p.flatten() for p in predictions]).numpy()
    all_gt = torch.cat([g.flatten() for g in ground_truths]).numpy()
    all_gt_binary = (all_gt > 0.5).astype(int)
    
    # Compute PR curve
    precision, recall, _ = precision_recall_curve(all_gt_binary, all_pred)
    
    # Baseline (random classifier)
    baseline = np.sum(all_gt_binary) / len(all_gt_binary)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(recall, precision, color='#2ecc71', linewidth=2,
            label=f'PR Curve (AUC-PR = {auc_pr:.4f})')
    ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.7,
               label=f'Baseline (prevalence = {baseline:.4f})')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    
    plt.tight_layout()
    
    for ext in ['png', 'pdf']:
        path = os.path.join(plots_dir, f'pr_curve.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
    
    plt.close(fig)


# ============================================================================
# Comparison Visualizations
# ============================================================================

def select_best_improvement_samples(
    predictions: List[torch.Tensor],
    ground_truths: List[torch.Tensor],
    optimal_threshold: float,
    num_samples: int = 4
) -> List[Tuple[int, float, float, float]]:
    """
    Select samples where optimal threshold improves Dice most vs 0.5.
    
    Returns:
        List of (index, improvement, dice_default, dice_optimal) tuples
    """
    metrics = get_all_metrics()
    dice_metric = metrics['dice']
    
    improvements = []
    
    for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        # Skip empty ground truths
        if gt.sum() == 0:
            continue
            
        dice_default = dice_metric(pred, gt, threshold=0.5)
        dice_optimal = dice_metric(pred, gt, threshold=optimal_threshold)
        improvement = dice_optimal - dice_default
        improvements.append((i, improvement, dice_default, dice_optimal))
    
    # Sort by improvement (descending)
    improvements.sort(key=lambda x: x[1], reverse=True)
    
    return improvements[:num_samples]


def generate_comparison_visualizations(
    predictions: List[torch.Tensor],
    ground_truths: List[torch.Tensor],
    modalities: List[torch.Tensor],
    optimal_threshold: float,
    output_dir: str,
    num_samples: int,
    cfg: DictConfig
):
    """Generate side-by-side comparison images."""
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Select best improvement samples
    selected = select_best_improvement_samples(
        predictions, ground_truths, optimal_threshold, num_samples
    )
    
    if not selected:
        print("  Warning: No samples with foreground found for visualization")
        return
    
    # Get modality names from config
    modality_names = cfg.dataset.modalities if hasattr(cfg.dataset, 'modalities') else []
    
    for rank, (idx, improvement, dice_default, dice_optimal) in enumerate(selected):
        pred = predictions[idx]
        gt = ground_truths[idx]
        mods = modalities[idx]
        
        # Calculate number of columns: modalities + GT + pred@0.5 + pred@optimal
        num_modalities = mods.shape[0]
        num_cols = num_modalities + 3
        
        fig, axes = plt.subplots(1, num_cols, figsize=(3 * num_cols, 3))
        
        # Plot modalities
        for m_idx in range(num_modalities):
            ax = axes[m_idx]
            img = mods[m_idx].numpy()
            ax.imshow(img, cmap='gray')
            if m_idx < len(modality_names):
                ax.set_title(modality_names[m_idx], fontsize=10)
            else:
                ax.set_title(f'Mod {m_idx+1}', fontsize=10)
            ax.axis('off')
        
        # Ground truth
        ax = axes[num_modalities]
        ax.imshow(gt[0].numpy(), cmap='Reds', vmin=0, vmax=1)
        ax.set_title('Ground Truth', fontsize=10)
        ax.axis('off')
        
        # Prediction @ 0.5
        ax = axes[num_modalities + 1]
        pred_05 = (pred[0] > 0.5).float().numpy()
        ax.imshow(pred_05, cmap='Blues', vmin=0, vmax=1)
        ax.set_title(f'τ=0.5\nDice={dice_default:.3f}', fontsize=10)
        ax.axis('off')
        
        # Prediction @ optimal
        ax = axes[num_modalities + 2]
        pred_opt = (pred[0] > optimal_threshold).float().numpy()
        ax.imshow(pred_opt, cmap='Greens', vmin=0, vmax=1)
        ax.set_title(f'τ={optimal_threshold:.2f}\nDice={dice_optimal:.3f}', fontsize=10)
        ax.axis('off')
        
        improvement_pct = (improvement / dice_default * 100) if dice_default > 0 else 0
        fig.suptitle(f'Sample {idx} | Improvement: {improvement_pct:+.1f}%', fontsize=12)
        
        plt.tight_layout()
        
        path = os.path.join(viz_dir, f'comparison_sample_{rank+1:03d}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    print(f"  Saved {len(selected)} visualizations to: {viz_dir}")


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_arguments()
    
    # Setup device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            args.run_dir, 'analysis', 'threshold_sweep', 
            f'{args.model_name}_{timestamp}'
        )
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    print("=" * 60)
    print("THRESHOLD ANALYSIS")
    print("=" * 60)
    print(f"  Run dir: {args.run_dir}")
    print(f"  Model: {args.model_name}")
    print(f"  Device: {device}")
    print(f"  Output: {output_dir}")
    if args.ensemble_samples >= 2:
        print(f"  Ensemble: {args.ensemble_samples} samples, method={args.ensemble_method}")
    else:
        print(f"  Ensemble: disabled (single sample)")
    print("=" * 60)
    
    # 1. Load config and model
    print("\n[1/8] Loading config...")
    cfg = load_config_from_run_dir(args.run_dir)
    
    print("[2/8] Loading model...")
    checkpoint_path = find_checkpoint(args.run_dir, args.model_name)
    print(f"  Checkpoint: {checkpoint_path}")
    model = load_model(cfg, checkpoint_path, device)
    
    # 2. Run inference
    print("\n[3/8] Running inference on validation set...")
    predictions, ground_truths, modalities = run_inference(model, cfg, device, args)
    num_samples = len(predictions)
    print(f"  Collected {num_samples} samples")
    
    # 3. Parse thresholds and compute sweep
    print("\n[4/8] Computing threshold sweep...")
    thresholds = parse_thresholds(args.thresholds)
    print(f"  Evaluating {len(thresholds)} thresholds: {thresholds[0]:.2f} to {thresholds[-1]:.2f}")
    results = compute_threshold_sweep(predictions, ground_truths, thresholds)
    
    # 4. Find optimal thresholds
    print("\n[5/8] Finding optimal thresholds...")
    optimal = find_optimal_thresholds(results, thresholds)
    print(f"  Optimal Dice: τ={optimal['dice']['threshold']:.2f} → {optimal['dice']['value']:.4f}")
    
    # 5. Compute AUC metrics
    print("\n[6/8] Computing AUC metrics...")
    auc_roc, auc_pr = compute_auc_metrics(predictions, ground_truths)
    print(f"  AUC-ROC: {auc_roc:.4f}")
    print(f"  AUC-PR:  {auc_pr:.4f}")
    
    # 6. Save outputs
    print("\n[7/8] Saving outputs...")
    save_csv_results(results, thresholds, output_dir, args.save_per_sample)
    save_json_summary(optimal, auc_roc, auc_pr, results, args, thresholds, num_samples, output_dir)
    save_summary_text(optimal, auc_roc, auc_pr, results, args, thresholds, num_samples, output_dir)
    
    # 7. Generate plots and visualizations
    print("\n[8/8] Generating plots and visualizations...")
    generate_plots(results, thresholds, optimal, auc_roc, auc_pr, 
                   predictions, ground_truths, output_dir)
    generate_comparison_visualizations(
        predictions, ground_truths, modalities,
        optimal['dice']['threshold'], output_dir, args.num_visualizations, cfg
    )
    
    print("\n" + "=" * 60)
    print("✓ Analysis complete!")
    print(f"  Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()

