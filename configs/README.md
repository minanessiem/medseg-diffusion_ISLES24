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

### Config Parameters

See `openai_base.yaml` for detailed parameter documentation.

## Composition and Usage
Run the project with Hydra, overriding groups as needed:
- Local run: `python main.py --config-name local`
- Cluster run with custom optimizer: `python main.py --config-name cluster optimizer=high_lr` (assuming high_lr.yaml exists in optimizer/).
- Test new diffusion: `python main.py --config-name local diffusion=openai_ddpm_100ts_debug` (to use OpenAI implementation).

Interpolation is used (e.g., ${environment.training.output_root} in training templates).

## Deprecations
- Old dataset files (isles24_local.yaml, isles24_cluster.yaml) are deprecated. Use `dataset: base_isles24` with the appropriate environment instead.
- Removed keys (e.g., learning_rate from training/) are now in optimizer/ or diffusion/.

For more on Hydra: https://hydra.cc/docs/intro/
