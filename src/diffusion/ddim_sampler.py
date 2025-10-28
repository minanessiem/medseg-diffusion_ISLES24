import torch
import torch.nn as nn

from .diffusion import Diffusion

class DDIMSampler(Diffusion):
    """
    Stub implementation for DDIM sampler, inheriting from abstract Diffusion.
    To be fully implemented in the future.
    """

    def __init__(self, model, cfg, device=None):
        super().__init__(model, cfg, device)
        # DDIM-specific setup will be added here

    def forward(self, mask, conditioned_image, return_intermediates=False, *args, **kwargs):
        raise NotImplementedError("DDIM forward pass not yet implemented")

    def sample(self, conditioned_image, disable_tqdm=False):
        raise NotImplementedError("DDIM sampling not yet implemented")

    def sample_with_snapshots(self, conditioned_image, snapshot_interval: int = None):
        raise NotImplementedError("DDIM sampling with snapshots not yet implemented")
