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
        """
        Factory method to create diffusion instances.
        
        Supports current custom implementations (DDPM, DDIM) and OpenAI-based
        implementation (OpenAI_DDPM supports both DDPM and DDIM sampling modes).
        
        Args:
            model (nn.Module): The base UNet (possibly wrapped in DataParallel)
            cfg (DictConfig): Hydra configuration
            device (torch.device, optional): Target device
            
        Returns:
            Diffusion: Instance of requested diffusion implementation
            
        Raises:
            ValueError: If diffusion type unknown
            ImportError: If OpenAI adapter unavailable
        
        Notes:
            - OpenAI_DDPM type supports both DDPM and DDIM sampling via the
              'sampling_mode' config parameter (ddpm/ddim)
            - For consistency, both openai_ddpm_*.yaml and openai_ddim_*.yaml
              configs use type: OpenAI_DDPM but differ in sampling_mode
        """
        OmegaConf.set_struct(cfg, False)
        diffusion_type = cfg.diffusion.type
        OmegaConf.set_struct(cfg, True)

        if diffusion_type == "DDPM":
            from .ddpm_sampler import DDPMSampler
            return DDPMSampler(model, cfg, device)
        elif diffusion_type == "DDIM":
            from .ddim_sampler import DDIMSampler
            return DDIMSampler(model, cfg, device)
        elif diffusion_type == "OpenAI_DDPM":
            try:
                from .openai_adapter import GaussianDiffusionAdapter
                return GaussianDiffusionAdapter(model, cfg, device)
            except ImportError as e:
                raise ImportError(
                    f"OpenAI diffusion adapter not available. "
                    f"Ensure src/improved_diffusion/ package is installed. "
                    f"Run the following to check: "
                    f"  python -c 'from src.improved_diffusion import GaussianDiffusion' "
                    f"Original error: {e}"
                )
        else:
            available_types = ["DDPM", "DDIM", "OpenAI_DDPM"]
            raise ValueError(
                f"Unknown diffusion type: '{diffusion_type}'. "
                f"Available types: {available_types}. "
                f"Note: OpenAI_DDPM supports both DDPM and DDIM sampling modes "
                f"via the 'sampling_mode' config parameter."
            )
