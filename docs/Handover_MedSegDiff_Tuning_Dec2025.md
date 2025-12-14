# MedSegDiff Tuning Handover Document

**Date**: December 14, 2025  
**Context**: Hyperparameter tuning and architecture comparison for ISLES24 stroke lesion segmentation  
**Dataset**: ISLES24 (CBF + TMAX modalities, 2D slices, fold 0)  
**Diffusion**: DDIM with 50 steps (respaced from 1000 timesteps), cosine noise schedule

---

## Executive Summary

This document captures the experimental progress on training diffusion-based segmentation models for ischemic stroke lesion detection. The session focused on:

1. **Analyzing training instabilities** across different MedSegDiff variants
2. **Fixing a critical AMP bug** where validation sampling ran in FP32 instead of BF16
3. **Systematic hyperparameter exploration** of learning rates, schedulers, and model sizes
4. **Identifying the MSE-Dice gap** as a key issue requiring auxiliary losses

**Current Best Result**: 7% 2D Dice (open-source MedSegDiff, 16c, with AMP BF16)

---

## Architecture Comparison

### Models Evaluated

| Architecture | Implementation | Key Characteristics | Status |
|-------------|----------------|---------------------|--------|
| **MedSegDiff (open-source)** | `src/models/MedSegDiff/` | Dual encoder, FFParser, DynamicFusionLayer, Transformer bottleneck | ✅ Best performer |
| **ORGMedSegDiff** | `src/models/ORGMedSegDiff/` | Single UNet + Highway network, calibration loss | ⚠️ Training instability |
| **DiffSwinTr** | `src/models/DiffSwinTr/` | Swin Transformer + CEM conditioning | 🔄 Not yet tested |

### Why Open-Source MedSegDiff Outperforms ORGMedSegDiff

1. **Dual encoder design** aligns with paper description: *"I and x_t are encoded with two individual encoders"*
2. **DynamicFusionLayer** with FFParser (Fourier attention) provides better feature fusion
3. **ORGMedSegDiff issues**:
   - Highway network hardcoded to `num_pool=5` causing depth mismatches
   - `.detach()` on anchor features decouples training signals
   - Calibration loss (weight=10) dominates over MSE
   - AMP instability in larger attention computations

---

## Model Configurations

### MedSegDiff Variants (Open-Source)

#### Small/Original (16c) - `medsegdiff_256_4l_16c_6x4a_128t_1btl_large.yaml`
```yaml
image_size: 256
num_layers: 4
first_conv_channels: 16        # Channel progression: 16→32→64→128
time_embedding_dim: 128
att_heads: 6
att_head_dim: 4                # Total attention dim: 24
bottleneck_transformer_layers: 1
```
- **Memory**: Fits batch_size=32 on single GPU with AMP BF16
- **Validation**: Can run val_batch_size=64
- **Best Dice**: 7% (with AMP BF16)

#### Medium-Simple (32c) - `medsegdiff_256_4l_32c_6x4a_128t_1btl_medium_simple.yaml`
```yaml
first_conv_channels: 32        # Channel progression: 32→64→128→256
time_embedding_dim: 128        # Keep original (stable)
att_heads: 6
att_head_dim: 4                # Keep original (stable)
bottleneck_transformer_layers: 1
```
- **Memory**: Fits batch_size=16 on single GPU with AMP BF16
- **Rationale**: Isolate channel capacity increase from attention changes
- **Status**: Pending evaluation

#### Medium-Full (32c) - `medsegdiff_256_4l_32c_8x8a_256t_2btl_medium.yaml`
```yaml
first_conv_channels: 32
time_embedding_dim: 256        # Richer conditioning
att_heads: 8
att_head_dim: 8                # Total attention dim: 64
bottleneck_transformer_layers: 2
```
- **Memory**: Fits batch_size=16 on single GPU with AMP BF16
- **Status**: ❌ NaN losses during training (numerical instability)
- **Likely cause**: 8×8 attention + BF16 precision issues

---

## Hyperparameter Configurations

### Optimizers

| Config | LR | Weight Decay | Use Case |
|--------|-----|--------------|----------|
| `adamw_1e4lr_wd00` | 1e-4 | 0.0 | Standard (16c models) |
| `adamw_5e5lr_wd00` | 5e-5 | 0.0 | Conservative (32c models, stability) |
| `adamw_2e4lr_wd00` | 2e-4 | 0.0 | Aggressive (not recommended) |

### Schedulers

| Config | Type | Details |
|--------|------|---------|
| `constant` | Constant LR | No decay - current baseline |
| `warmup_cosine_10pct` | Warmup + Cosine | 10% warmup, cosine to 0 - **recommended** |
| `warmup_cosine_5pct` | Warmup + Cosine | 5% warmup, cosine to 0 |

### Loss Configurations

| Config | Losses | Notes |
|--------|--------|-------|
| `mse_loss_only` | MSE only | Current baseline |
| `mse_with_aux_light` | MSE + 0.1×Dice + 0.1×BCE | 10K warmup, **recommended for Dice gap** |
| `mse_with_calibration` | MSE + 10×CalibrationBCE | For ORGMedSegDiff highway network |

### Training Configs

| Config | AMP | Steps | Notes |
|--------|-----|-------|-------|
| `train_100Kms_ampbfloat16` | BF16 | 100K default | **Use this + override max_steps** |

---

## Critical Bug Fix: AMP in Sampling

### Issue
Validation sampling (`diffusion.sample()`) was running in FP32 even when training used BF16, causing:
- OOM during validation with reasonable batch sizes
- Memory mismatch between training and inference

### Fix Applied
**File**: `src/diffusion/openai_adapter.py`

1. Store AMP config in `__init__`:
```python
amp_cfg = cfg.training.get('amp', {})
self.amp_enabled = amp_cfg.get('enabled', False)
self.amp_dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16, ...}
```

2. Apply autocast in `sample()` and `sample_with_snapshots()`:
```python
from torch.amp import autocast
with torch.no_grad(), autocast(device_type='cuda', enabled=self.amp_enabled, dtype=self.amp_dtype):
    # DDIM/DDPM sampling
```

### Impact
- Training batch=32 + Validation batch=64 now fits on single GPU
- ~40% memory reduction during validation sampling

---

## Experimental Results

### Completed Runs

| Run | Model | LR | Scheduler | Dice | Notes |
|-----|-------|-----|-----------|------|-------|
| 16c baseline | 16c | 1e-4 | constant | 6.5% | No AMP |
| 16c + AMP BF16 | 16c | 1e-4 | constant | **7.0%** | Best so far |
| 32c-full | 32c (8×8 attn) | 1e-4 | constant | NaN | Crashed |

### Key Observations

1. **Dice Oscillation**: Scores oscillate between 2-7% without clear upward trend
2. **"Snow" Noise**: Final denoising step produces noisy predictions
3. **MSE-Dice Gap**: Low MSE loss doesn't correlate with high Dice
4. **Training Instability**: Larger models prone to NaN with current settings

---

## Pending Experiments

### Priority 1: Auxiliary Losses (Address MSE-Dice Gap)
```bash
python3 -m scripts.slurm.single_job_runner --config-name cluster_openai_ddim50 --gpus 1 \
  --overrides model=medsegdiff_256_4l_16c_6x4a_128t_1btl_large \
  loss=mse_with_aux_light \
  augmentation=aggressive_2d \
  scheduler=warmup_cosine_10pct \
  optimizer=adamw_1e4lr_wd00 \
  training=train_100Kms_ampbfloat16 \
  training.max_steps=500000 \
  environment.dataset.train_batch_size=32 \
  environment.dataset.num_valid_workers=64 \
  validation.val_batch_size=64
```

### Priority 2: 32c-Simple + Lower LR (Test Capacity Safely)
```bash
python3 -m scripts.slurm.single_job_runner --config-name cluster_openai_ddim50 --gpus 1 \
  --overrides model=medsegdiff_256_4l_32c_6x4a_128t_1btl_medium_simple \
  loss=mse_loss_only \
  augmentation=aggressive_2d \
  scheduler=warmup_cosine_10pct \
  optimizer=adamw_5e5lr_wd00 \
  training=train_100Kms_ampbfloat16 \
  training.max_steps=500000 \
  environment.dataset.train_batch_size=16 \
  environment.dataset.num_valid_workers=64 \
  validation.val_batch_size=64
```

### Priority 3: 32c-Simple + Aux Losses (Combined Attack)
```bash
python3 -m scripts.slurm.single_job_runner --config-name cluster_openai_ddim50 --gpus 1 \
  --overrides model=medsegdiff_256_4l_32c_6x4a_128t_1btl_medium_simple \
  loss=mse_with_aux_light \
  augmentation=aggressive_2d \
  scheduler=warmup_cosine_10pct \
  optimizer=adamw_5e5lr_wd00 \
  training=train_100Kms_ampbfloat16 \
  training.max_steps=500000 \
  environment.dataset.train_batch_size=16 \
  environment.dataset.num_valid_workers=64 \
  validation.val_batch_size=64
```

### LR/Scheduler Ablation (16c model)
Already submitted:
- 5e-5 LR + constant
- 5e-5 LR + warmup_cosine_10pct  
- 1e-4 LR + warmup_cosine_10pct

---

## Command Templates

### Local Testing (with early validation)
```bash
python main.py --config-name cluster_openai_ddim50 \
  model=<model_config> \
  loss=<loss_config> \
  augmentation=aggressive_2d \
  scheduler=<scheduler_config> \
  optimizer=<optimizer_config> \
  training=train_100Kms_ampbfloat16 \
  training.max_steps=<steps> \
  environment.dataset.train_batch_size=<batch> \
  validation.validation_interval=200  # Early validation for memory testing
```

### SLURM Submission
```bash
python3 -m scripts.slurm.single_job_runner \
  --config-name cluster_openai_ddim50 \
  --gpus <num_gpus> \
  --cpus-per-task <cpus> \
  --overrides \
    model=<model_config> \
    loss=<loss_config> \
    ... # other overrides
```

### Multi-GPU (auto-configured)
When using `--gpus 2`, the runner automatically adds:
```
environment.training.multi_gpu=[0,1]
```

---

## Run Name Convention

Run names are auto-generated by `src/utils/run_name.py`:

```
{model}_{batch}_{amp}_{optimizer}_{scheduler}_{steps}_{loss}_{aug}_{diffusion}_{timestamp}
```

Example:
```
medsegdiff_256_4l_16c_6x4a_128t_1btl_b32_ampBF16_adamw1e4_wd00_wcos10_s500K_lMSE_dw10_d01_b01_w10K_augAGG2D_oai_ddim_ds1000_nzcosine_tr50_2025-12-14_15-30-00
```

Key components:
- `b32` - batch size 32
- `ampBF16` - AMP with bfloat16
- `adamw1e4_wd00` - AdamW, 1e-4 LR, no weight decay
- `wcos10` - warmup_cosine with 10% warmup
- `s500K` - 500K steps
- `lMSE_dw10_d01_b01_w10K` - MSE with aux (diffusion_weight=1.0, dice=0.1, bce=0.1, warmup=10K)

---

## Next Steps If Current Approach Fails

If auxiliary losses + capacity increases don't break 15-20% Dice:

1. **DiffSwinTr**: Already implemented in `src/models/DiffSwinTr/`
   - Swin Transformer for better fine-grained details
   - CEM (Conditional Encoder Module) for conditioning
   - Start with `diffswintr_s.yaml`

2. **Data Investigation**:
   - Check class balance in training data
   - Visualize hard examples
   - Consider per-patient aggregation vs slice-wise

3. **Diffusion Settings**:
   - Try DDIM with more steps (100, 250)
   - Experiment with eta > 0 for stochasticity
   - Consider DDPM (1000 steps) for final quality

---

## Files Modified This Session

1. **`src/diffusion/openai_adapter.py`** - Added AMP autocast to sampling
2. **`configs/model/medsegdiff_256_4l_32c_8x8a_256t_2btl_medium.yaml`** - Created medium model
3. **`configs/loss/mse_with_aux_light.yaml`** - Created auxiliary loss config

---

## Contact/Context

This work is part of exploring DDPM/DDIM diffusion models for multi-modal CT scan segmentation for ischemic stroke lesions using the ISLES24 dataset. The codebase combines:

- Open-source MedSegDiff implementation (faithful to paper)
- OpenAI's improved_diffusion package (battle-tested diffusion code)
- Hydra configuration system
- SLURM job submission infrastructure

The goal is to achieve competitive Dice scores (40-60% target based on literature) through systematic architecture and hyperparameter exploration.

