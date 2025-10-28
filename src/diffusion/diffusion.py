import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from omegaconf import OmegaConf

class Diffusion(nn.Module, ABC):
    """
    Abstract base class for diffusion models, defining interfaces used in training.
    """

    def __init__(self, model, cfg, device=None):
        super().__init__()
        self.model = model
        self.image_channels = model.image_channels
        self.mask_channels = model.mask_channels
        self.image_size = model.image_size
        self.device = torch.device(cfg.environment.device) if device is None else device

    @abstractmethod
    def forward(self, mask, conditioned_image, return_intermediates=False, *args, **kwargs):
        pass

    @abstractmethod
    def sample(self, conditioned_image, disable_tqdm=False):
        pass

    @abstractmethod
    def sample_with_snapshots(self, conditioned_image, snapshot_interval: int = None):
        pass

    @classmethod
    def build_diffusion(cls, model, cfg, device=None):
        OmegaConf.set_struct(cfg, False)
        diffusion_type = cfg.diffusion.type
        OmegaConf.set_struct(cfg, True)

        if diffusion_type == "DDPM":
            from .ddpm_sampler import DDPMSampler
            return DDPMSampler(model, cfg, device)
        elif diffusion_type == "DDIM":
            from .ddim_sampler import DDIMSampler
            return DDIMSampler(model, cfg, device)
        else:
            raise ValueError(f"Unknown diffusion type: {diffusion_type}")
