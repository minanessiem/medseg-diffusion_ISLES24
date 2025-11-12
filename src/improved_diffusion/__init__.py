"""
OpenAI improved_diffusion core components (math only).

This package contains the core diffusion mathematics from OpenAI's improved-diffusion:
https://github.com/openai/improved-diffusion

Files are copied without modification to enable easy upstream pulls.
See LICENSE_OPENAI for attribution.

Components included:
- gaussian_diffusion: Core DDPM/DDIM forward/reverse processes
- respace: Spaced diffusion for fewer sampling steps
- nn: Helper functions (update_ema, mean_flat)
- losses: Loss functions (KL divergence, discretized Gaussian log-likelihood)
- resample: Timestep sampling strategies (uniform, loss-aware)

Components excluded (not needed):
- train_util.py: We use our own trainer
- logger.py: We use our own logger
- unet.py: We use our own conditional UNet
- image_datasets.py: We use our own data loaders
- dist_util.py: We use DataParallel, not DDP
- fp16_util.py: Deferred to future work
- script_util.py: Not needed
"""

from .gaussian_diffusion import (
    GaussianDiffusion,
    get_named_beta_schedule,
    ModelMeanType,
    ModelVarType,
    LossType,
)
from .respace import SpacedDiffusion, space_timesteps
from .nn import update_ema, mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood
from .resample import UniformSampler, LossAwareSampler, LossSecondMomentResampler

__all__ = [
    "GaussianDiffusion",
    "SpacedDiffusion",
    "space_timesteps",
    "get_named_beta_schedule",
    "ModelMeanType",
    "ModelVarType",
    "LossType",
    "update_ema",
    "mean_flat",
    "normal_kl",
    "discretized_gaussian_log_likelihood",
    "UniformSampler",
    "LossAwareSampler",
    "LossSecondMomentResampler",
]
