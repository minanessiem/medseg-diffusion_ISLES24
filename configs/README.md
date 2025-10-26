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

## Composition and Usage
Run the project with Hydra, overriding groups as needed:
- Local run: `python main.py --config-name local`
- Cluster run with custom optimizer: `python main.py --config-name cluster optimizer=high_lr` (assuming high_lr.yaml exists in optimizer/).
- Test new diffusion: `python main.py --config-name local diffusion=linear_200` (if linear_200.yaml is added).

Interpolation is used (e.g., ${environment.training.output_root} in training templates).

## Deprecations
- Old dataset files (isles24_local.yaml, isles24_cluster.yaml) are deprecated. Use `dataset: base_isles24` with the appropriate environment instead.
- Removed keys (e.g., learning_rate from training/) are now in optimizer/ or diffusion/.

For more on Hydra: https://hydra.cc/docs/intro/
