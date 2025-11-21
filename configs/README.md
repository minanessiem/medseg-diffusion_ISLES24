# Configuration Documentation

## Overview
This directory contains Hydra configuration files for the project, organized for modularity and composability. Key groups include:
- **environment/**: Environment-specific settings (e.g., paths, num_workers, device). Variants: local.yaml, cluster.yaml.
- **dataset/**: Dataset configurations. Use base_isles24.yaml for shared settings, composed with environment.
- **model/**: Model architectures (e.g., unet_local.yaml, unet_cluster.yaml).
- **training/**: Training loop parameters (e.g., max_steps, checkpoints). Variants: train_local.yaml, train_cluster.yaml (now slimmed down).
- **optimizer/**: Optimizer and scheduler settings (e.g., default.yaml with learning_rate, reduce_lr).
- **diffusion/**: Diffusion-specific params (e.g., default.yaml with timesteps, noise_schedule).
- **validation/** and **logging/**: As before.

Top-level files like cluster.yaml and local.yaml compose these groups via defaults lists.

## Diffusion Backends

### Current Implementation (Custom)
- `diffusion_100ts_cosinesch.yaml` - Custom DDPM (100 steps, cosine)

### OpenAI Implementation
Based on OpenAI's improved-diffusion package.

**Note:** All OpenAI configs use `type: OpenAI_DDPM` but support both DDPM and DDIM sampling modes via the `sampling_mode` parameter. This matches OpenAI's architecture where both are sampling strategies from the same `GaussianDiffusion` class.

**DDPM (Standard):**
- `openai_ddpm_1000ts_cosine.yaml` - Full 1000-step DDPM (`sampling_mode: ddpm`)
- `openai_ddpm_100ts_debug.yaml` - Fast 100-step for debugging (`sampling_mode: ddpm`)

**DDIM (Fast Sampling):**
- `openai_ddim_50steps.yaml` - 50-step DDIM (20x faster sampling, `sampling_mode: ddim`)
- `openai_ddim_250steps.yaml` - 250-step DDIM (4x faster sampling, `sampling_mode: ddim`)

**Experimental:**
- `openai_loss_aware.yaml` - Loss-aware sampling

### Choosing a Config

**For training:**
- Development/debugging: `openai_ddpm_100ts_debug`
- Production: `openai_ddpm_1000ts_cosine`

**For sampling (inference):**
- Fast preview: `openai_ddim_50steps` (50 steps)
- Balanced: `openai_ddim_250steps` (250 steps)
- Best quality: `openai_ddpm_1000ts_cosine` (1000 steps)

### Logging Integration

Each diffusion config automatically includes optimized logging settings:
- **Snapshot interval** is adjusted to show **10 evenly spaced** snapshots during sampling
- For DDPM (100 steps): interval=10 → snapshots at t=100, 90, 80, ..., 10, 0
- For DDPM (1000 steps): interval=100 → snapshots at t=1000, 900, 800, ..., 100, 0
- For DDIM (50 steps): interval=5 → snapshots at 10 evenly spaced respaced timesteps
- For DDIM (250 steps): interval=25 → snapshots at 10 evenly spaced respaced timesteps

**Note:** DDIM respacing means the actual sampling happens at fewer discrete timesteps. The logging configs account for this to ensure you see the full denoising progression.

### Config Parameters

See `openai_base.yaml` for detailed parameter documentation.

## Composition and Usage
Run the project with Hydra, overriding groups as needed:
- Local run: `python main.py --config-name local`
- Cluster run with custom optimizer: `python main.py --config-name cluster optimizer=high_lr` (assuming high_lr.yaml exists in optimizer/).
- Test new diffusion: `python main.py --config-name local diffusion=openai_ddpm_100ts_debug` (to use OpenAI implementation).

Interpolation is used (e.g., ${environment.training.output_root} in training templates).

## Checkpoint Configuration

The training system supports two independent checkpoint strategies:

### Interval Checkpoints
Periodic safety backups for training resumption. Configured in `training/checkpoint_interval`:

- `enabled`: Enable/disable interval checkpointing (boolean)
- `save_interval`: Steps between saves (e.g., 5000, 10000)
- `keep_last_n`: Retain only N most recent checkpoints via FIFO policy (null = keep all)
- `model_template`: Filename template for model checkpoint
- `opt_template`: Filename template for optimizer state

**Use case**: Fault recovery, training resumption after interruptions

**Saved artifacts**: Model state dict + optimizer state dict

### Best Model Checkpoints
Quality-gated saves based on validation metrics. Configured in `training/checkpoint_best`:

- `enabled`: Enable/disable metric-based checkpointing (boolean)
- `metric_name`: Validation metric to track (e.g., "dice_2d_fg", "f1_2d")
  - **Important**: Use the metric key from validation results (no "val_" prefix)
- `metric_mode`: "max" (higher is better) or "min" (lower is better)
- `keep_last_n`: Retain only top N checkpoints by metric value (null = keep all)
- `model_template`: Filename template for model checkpoint (includes metric value)
- `ema_template`: Filename template for EMA checkpoint (includes metric value)

**Use case**: Inference, model selection, experiment comparison

**Saved artifacts**: Model state dict + all configured EMA state dicts

### Configuration Examples

**Minimal (interval only)**:
```yaml
checkpoint_interval:
  enabled: true
  save_interval: 10000
  keep_last_n: 2
checkpoint_best:
  enabled: false
```

**Quality-focused (best models)**:
```yaml
checkpoint_interval:
  enabled: true
  save_interval: 10000
  keep_last_n: 3
checkpoint_best:
  enabled: true
  metric_name: "dice_2d_fg"
  metric_mode: "max"
  keep_last_n: 5
```

**Track loss instead**:
```yaml
checkpoint_best:
  enabled: true
  metric_name: "test_loss"
  metric_mode: "min"
  keep_last_n: 3
```

### File Organization

Checkpoints are saved to subdirectories under the run output directory:
```
outputs/run_name_timestamp/
├── models/
│   ├── checkpoint/          # Interval checkpoints (FIFO retention)
│   │   ├── diffusion_chkpt_step_095000.pth
│   │   ├── opt_chkpt_step_095000.pth
│   │   └── ...
│   └── best/                # Best model checkpoints (quality-based retention)
│       ├── best_model_step_052000_dice_2d_fg_0.8456.pth
│       ├── best_model_step_052000_dice_2d_fg_0.8456_ema_0.9999.pth
│       └── ...
```

### Checkpoint Logging

The system provides clear logging for all checkpoint decisions:
- `✓ Saving interval checkpoint at step X`
- `✓ Saving best model: dice_2d_fg improved 0.8234 → 0.8456`
- `✗ Skipping best model save: dice_2d_fg did not improve (current: 0.8234, best: 0.8456)`
- `🗑️ Removed old interval checkpoint: diffusion_chkpt_step_005000.pth`
- `🗑️ Removed worse best checkpoint: best_model_step_010000_dice_0.8001.pth`

### Error Handling

If `metric_name` doesn't exist in validation results, training will crash with a clear error message:
```
❌ ERROR: Checkpoint metric 'dice_2d' not found in validation results.
   Validation returned metrics: ['dice_2d_fg', 'f1_2d', 'precision_2d', ...]
   Check your checkpoint_best.metric_name config!
```

## Deprecations
- Old dataset files (isles24_local.yaml, isles24_cluster.yaml) are deprecated. Use `dataset: base_isles24` with the appropriate environment instead.
- Removed keys (e.g., learning_rate from training/) are now in optimizer/ or diffusion/.
- **As of 2025-11-21**: `checkpoint_save_interval`, `main_checkpoint_template`, `ema_checkpoint_template`, `opt_checkpoint_template` are deprecated. Use `checkpoint_interval` and `checkpoint_best` config groups instead.

For more on Hydra: https://hydra.cc/docs/intro/
