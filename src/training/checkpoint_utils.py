"""Checkpoint decision logic for training.

This module contains pure functions for determining when to save checkpoints
based on validation metrics or training intervals. It does not perform file I/O.
"""

import math
import os
import csv
import torch
from typing import Dict, List, Tuple


def should_save_best_checkpoint(
    val_results: Dict[str, float],
    best_metric_value: float,
    config
) -> Tuple[bool, str, float]:
    """
    Determine if best model checkpoint should be saved based on validation metric.
    
    Args:
        val_results: Dictionary of validation metrics from validate_one_epoch()
        best_metric_value: Current best metric value (-inf for max mode, +inf for min mode)
        config: cfg.training.checkpoint_best (DictConfig)
    
    Returns:
        Tuple of (should_save, reason, new_best_value):
            - should_save (bool): Whether to save checkpoint
            - reason (str): Human-readable explanation for logging
            - new_best_value (float): Updated best metric value
    
    Raises:
        KeyError: If config.metric_name not found in val_results
    
    Examples:
        >>> from types import SimpleNamespace
        >>> val_results = {'dice_2d_fg': 0.85, 'f1_2d': 0.82}
        >>> config = SimpleNamespace(metric_name='dice_2d_fg', metric_mode='max')
        >>> should_save_best_checkpoint(val_results, float('-inf'), config)
        (True, "Saving first best model: dice_2d_fg = 0.8500 (baseline)", 0.85)
    """
    metric_name = config.metric_name
    metric_mode = config.metric_mode
    
    # Check metric exists
    if metric_name not in val_results:
        available = list(val_results.keys())
        raise KeyError(
            f"Checkpoint metric '{metric_name}' not found in validation results. "
            f"Available metrics: {available}"
        )
    
    # Convert tensor to float for comparison (metrics return torch.Tensor)
    current_value = float(val_results[metric_name])
    
    # Handle NaN/inf gracefully
    if not math.isfinite(current_value):
        reason = (
            f"Skipping best model save: {metric_name} = {current_value} (not finite)"
        )
        return False, reason, best_metric_value
    
    # First validation check (initial best_metric_value is sentinel)
    is_first = (
        (metric_mode == 'max' and best_metric_value == float('-inf')) or
        (metric_mode == 'min' and best_metric_value == float('inf'))
    )
    
    if is_first:
        reason = f"Saving first best model: {metric_name} = {current_value:.4f} (baseline)"
        return True, reason, current_value
    
    # Check improvement
    improved = (
        (metric_mode == 'max' and current_value > best_metric_value) or
        (metric_mode == 'min' and current_value < best_metric_value)
    )
    
    if improved:
        reason = (
            f"Saving best model: {metric_name} improved "
            f"{best_metric_value:.4f} → {current_value:.4f}"
        )
        return True, reason, current_value
    else:
        reason = (
            f"Skipping best model save: {metric_name} did not improve "
            f"(current: {current_value:.4f}, best: {best_metric_value:.4f})"
        )
        return False, reason, best_metric_value


def save_interval_checkpoint(
    diffusion,
    optimizer,
    step: int,
    cfg,
    run_dir: str
) -> List[str]:
    """
    Save interval checkpoint (model + optimizer) for training resumption.
    
    Args:
        diffusion: Diffusion model (with state_dict method)
        optimizer: Optimizer (with state_dict method)
        step: Current training step
        cfg: Hydra config (must have cfg.training.checkpoint_interval)
        run_dir: Root directory for saving (e.g., "outputs/run_name/")
    
    Returns:
        List of saved file paths (for tracking/cleanup)
    
    Raises:
        OSError: If file save fails
    """
    saved_files = []
    config = cfg.training.checkpoint_interval
    
    # Save model
    model_path = f"{run_dir}{config.model_template.format(step)}"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(diffusion.state_dict(), model_path)
    saved_files.append(model_path)
    print(f"  ├─ Model: {model_path}")
    
    # Save optimizer
    opt_path = f"{run_dir}{config.opt_template.format(step)}"
    os.makedirs(os.path.dirname(opt_path), exist_ok=True)
    torch.save(optimizer.state_dict(), opt_path)
    saved_files.append(opt_path)
    print(f"  └─ Optimizer: {opt_path}")
    
    return saved_files


def save_best_checkpoint(
    diffusion,
    ema_params: List,
    ema_rates: List[float],
    step: int,
    cfg,
    run_dir: str,
    val_results: Dict[str, float]
) -> List[str]:
    """
    Save best model checkpoint (model + EMAs + metrics CSV) for inference.
    
    Args:
        diffusion: Diffusion model (with state_dict method)
        ema_params: List of EMA parameter copies (one per rate)
        ema_rates: List of EMA decay rates
        step: Current training step
        cfg: Hydra config (must have cfg.training.checkpoint_best and cfg.training.ema_rate_precision)
        run_dir: Root directory for saving
        val_results: Validation metrics dictionary
    
    Returns:
        List of saved file paths (for tracking/cleanup)
    
    Raises:
        OSError: If file save fails
        KeyError: If metric not in val_results
    """
    saved_files = []
    config = cfg.training.checkpoint_best
    metric_name = config.metric_name
    # Convert tensor to float for filename formatting
    metric_value = float(val_results[metric_name])
    
    # Format kwargs for templates
    format_kwargs = {
        'step': step,
        'metric_name': metric_name,
        'metric_value': metric_value
    }
    
    # Save main model
    model_path = f"{run_dir}{config.model_template.format(**format_kwargs)}"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(diffusion.state_dict(), model_path)
    saved_files.append(model_path)
    print(f"  ├─ Model: {model_path}")
    
    # Save EMAs (if configured)
    if ema_params and ema_rates:
        for rate, params in zip(ema_rates, ema_params):
            formatted_rate = f"{rate:.{cfg.training.ema_rate_precision}f}"
            format_kwargs['rate'] = formatted_rate
            ema_path = f"{run_dir}{config.ema_template.format(**format_kwargs)}"
            
            # Build EMA state dict
            ema_state = {
                k: v for k, v in zip(
                    diffusion.state_dict().keys(),
                    (p.data for p in params)
                )
            }
            torch.save(ema_state, ema_path)
            saved_files.append(ema_path)
            print(f"  ├─ EMA ({rate}): {ema_path}")
    
    # Save metrics CSV (if enabled)
    if config.get('save_metrics_csv', False):
        csv_path = f"{run_dir}{config.metrics_template.format(**format_kwargs)}"
        write_metrics_csv(csv_path, val_results, precision=6)
        saved_files.append(csv_path)
        print(f"  ├─ Metrics CSV: {csv_path}")
    
    if saved_files:
        # Update last line to use └─ for final entry
        print(f"  └─ Total files: {len(saved_files)}")
    
    return saved_files


def cleanup_interval_checkpoints(
    checkpoint_list: List[Tuple[int, List[str]]],
    keep_last_n: int
) -> List[Tuple[int, List[str]]]:
    """
    Remove oldest interval checkpoints, keeping only last N.
    
    This function implements a FIFO retention policy based on training step.
    It deletes files immediately after identifying them for removal.
    
    Args:
        checkpoint_list: List of (step, [file_paths]) tuples
        keep_last_n: Number of most recent checkpoints to retain
    
    Returns:
        Updated checkpoint_list with old entries removed
    
    Notes:
        - Silently skips files that don't exist (already deleted)
        - Sorts by step to ensure FIFO ordering
    """
    if keep_last_n is None or keep_last_n <= 0:
        return checkpoint_list
    
    if len(checkpoint_list) <= keep_last_n:
        return checkpoint_list
    
    # Sort by step (ascending) to ensure FIFO
    checkpoint_list.sort(key=lambda x: x[0])
    
    # Identify checkpoints to remove
    to_remove = checkpoint_list[:len(checkpoint_list) - keep_last_n]
    
    # Delete files
    for step, file_paths in to_remove:
        for fpath in file_paths:
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                    print(f"🗑️  Removed old interval checkpoint: {os.path.basename(fpath)}")
                except OSError as e:
                    print(f"⚠️  Warning: Could not remove {fpath}: {e}")
    
    # Return only kept checkpoints
    return checkpoint_list[-keep_last_n:]


def cleanup_best_checkpoints(
    checkpoint_list: List[Tuple[float, int, List[str]]],
    keep_last_n: int,
    metric_mode: str
) -> List[Tuple[float, int, List[str]]]:
    """
    Remove worst best-model checkpoints, keeping only top N by metric value.
    
    This function implements a quality-based retention policy, keeping the
    N best checkpoints according to the metric mode (max or min).
    
    Args:
        checkpoint_list: List of (metric_value, step, [file_paths]) tuples
        keep_last_n: Number of best checkpoints to retain
        metric_mode: "max" (higher is better) or "min" (lower is better)
    
    Returns:
        Updated checkpoint_list with worst entries removed
    
    Notes:
        - For "max" mode: keeps highest metric values
        - For "min" mode: keeps lowest metric values
        - Silently skips files that don't exist
    """
    if keep_last_n is None or keep_last_n <= 0:
        return checkpoint_list
    
    if len(checkpoint_list) <= keep_last_n:
        return checkpoint_list
    
    # Sort by metric value
    # For max mode: descending (best first)
    # For min mode: ascending (best first)
    reverse = (metric_mode == 'max')
    checkpoint_list.sort(key=lambda x: x[0], reverse=reverse)
    
    # Identify checkpoints to remove (worst performers)
    to_remove = checkpoint_list[keep_last_n:]
    
    # Delete files
    for metric_val, step, file_paths in to_remove:
        for fpath in file_paths:
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                    print(f"🗑️  Removed worse best checkpoint: {os.path.basename(fpath)}")
                except OSError as e:
                    print(f"⚠️  Warning: Could not remove {fpath}: {e}")
    
    # Return only kept checkpoints (top N)
    return checkpoint_list[:keep_last_n]


def write_metrics_csv(
    file_path: str,
    metrics_dict: Dict[str, float],
    precision: int = 6
) -> None:
    """
    Write validation metrics to a CSV file with header row.
    
    Args:
        file_path: Path to save CSV file
        metrics_dict: Dictionary of metric_name -> metric_value
        precision: Number of decimal places for floating point values
    
    Raises:
        OSError: If file write fails
    
    Example output:
        metric_key,metric_value
        dice_2d_fg,0.845632
        f1_2d,0.823451
        precision_2d,0.891234
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['metric_key', 'metric_value'])
        # Write metrics (sorted for consistency)
        for key in sorted(metrics_dict.keys()):
            value = float(metrics_dict[key])  # Convert tensor to float if needed
            formatted_value = f"{value:.{precision}f}"
            writer.writerow([key, formatted_value])


