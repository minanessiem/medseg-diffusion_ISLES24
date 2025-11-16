import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.models.MedSegDiff.unet_util import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one, default, identity, ModelPrediction
from src.utils.general import device_grad_decorator
from omegaconf import OmegaConf

from .diffusion import Diffusion
from .noise_scheduler import NoiseScheduler  # Will be implemented separately

class DDPMSampler(Diffusion):
    """
    DDPM-specific implementation of the diffusion sampler, inheriting from abstract Diffusion.
    A denoising probabilistic Gaussian diffusion model (DDPM).

    Attributes:
        model (nn.Module): Neural network model for generating predictions.
        image_channels (int): Number of input image channels.
        mask_channels (int): Number of mask channels.
        image_size (int): Size of the image.
        device (torch.device): Device to run the model (CPU or GPU).
        num_timesteps (int): Number of timesteps for the diffusion process.
        Various other attributes for the diffusion process.
    """

    def __init__(self, model, cfg, device=None):
        """
        Initialize the DDPM sampler.

        Args:
            model (nn.Module): Neural network model for generating predictions.
            cfg (DictConfig): Hydra configuration object.
            device (torch.device): Device to run the model (CPU or GPU).
        """
        super().__init__(model, cfg, device)
        OmegaConf.set_struct(cfg, False)
        # Temporary aliases for config transition
        cfg.training.timesteps = cfg.diffusion.timesteps
        cfg.training.noise_schedule = cfg.diffusion.noise_schedule
        OmegaConf.set_struct(cfg, True)

        self._setup_diffusion_parameters(cfg.training.timesteps, cfg.training.noise_schedule)

    def _setup_diffusion_parameters(self, timesteps, noise_schedule):
        """
        Setup the diffusion process parameters for DDPM.

        Args:
            timesteps (int): Number of timesteps for the diffusion process.
            noise_schedule (str): Type of noise scheduling.
        """
        noise_scheduler = NoiseScheduler(timesteps, mode=noise_schedule)
        betas = torch.tensor(noise_scheduler.get_beta_schedule())
        alphas = torch.tensor(noise_scheduler.get_alpha_schedule())
        alpha_bars = torch.tensor(noise_scheduler.get_alpha_bar_schedule())
        alpha_bars_prev = torch.tensor(noise_scheduler.get_alphas_bar_previous_schedule())

        self.num_timesteps = betas.shape[0]
        self._setup_diffusion_tensors(betas, alphas, alpha_bars, alpha_bars_prev)

    def _setup_diffusion_tensors(self, betas, alphas, alpha_bars, alpha_bars_prev):
        """
        Setup and convert diffusion tensors to the correct type and device, and register them as buffers for DDPM.

        Args:
            betas, alphas, alpha_bars, alpha_bars_prev: Tensors for diffusion.
        """
        #       register a buffer for the following parameters for later use ( pts):
        #       beta(t), log(beta(t)), alpha_bar(t), alpha_bar(t-1), sqrt(alpha_bar(t)), sqrt(1 - alpha_bar(t)),
        #       1 / sqrt(alpha(t)), 1 / sqrt(alpha_bar(t)), beta(t) / sqrt(1 - alpha_bar(t)), beta_tilda(t), log(beta_tilda(t)).

        beta_tilda = betas * (1. - alpha_bars_prev) / (1. - alpha_bars)
        self.register_buffer('betas', betas.float())

        self.register_buffer('alpha_bars', alpha_bars.float())
        self.register_buffer('alpha_bars_prev', alpha_bars_prev.float())
        self.register_buffer('sqrt_alpha_bars', torch.sqrt(alpha_bars).float())
        self.register_buffer('sqrt_one_minus_alpha_bars', torch.sqrt(1. - alpha_bars).float())
        self.register_buffer('sqrt_inverse_alpha_bars', torch.sqrt(1. / alpha_bars).float())

        self.register_buffer('sqrt_inverse_alphas', torch.sqrt(1. / alphas).float())
        self.register_buffer('estimated_mean_noise_coefficient', (betas / torch.sqrt(1. - alpha_bars)).float())

        self.register_buffer('estimated_log_variance_beta_tilda', torch.log(beta_tilda.clamp(min=1e-20)).float())
        self.register_buffer('estimated_log_variance_beta', torch.log(betas).float())

    @device_grad_decorator(device=None, no_grad=True)  # Use from src/utils/general
    def batch_select_time_indices(self, tensor, indices):
        """
        Args:
            tensor (torch.Tensor): The input tensor from which elements are to be extracted.
            indices (torch.Tensor): The tensor containing indices at which to extract elements.
        Returns:
            torch.Tensor: The reshaped tensor after extraction with shape [batch_size, 1, 1, 1]
        """
        return tensor.gather(dim=-1, index=indices).reshape(indices.shape[0], 1, 1, 1)

    @device_grad_decorator(device=None, no_grad=True)
    def calculate_x0_from_xt(self, x_t, t, noise):
        """
        Predict the start image from noise in DDPM.

        Args:
            x_t (tensor): The noisy image at timestep t.
            t (int): The current timestep.
            noise (tensor): The noise tensor.

        Returns:
            The predicted start image at t=0.
        """
        sqrt_inverse_alpha_bars = self.batch_select_time_indices(self.sqrt_inverse_alpha_bars, t)
        sqrt_one_minus_alpha_bar_t = self.batch_select_time_indices(self.sqrt_one_minus_alpha_bars, t)
        return (x_t - sqrt_one_minus_alpha_bar_t * noise) * sqrt_inverse_alpha_bars

    @device_grad_decorator(device=None, no_grad=True)
    def estimate_mean(self, predicted_noise, x_t, t):
        """
        Estimate the mean for the reverse process in DDPM.

        Args:
            predicted_noise (tensor): Predicted noise.
            x_t (tensor): Noisy image at timestep t.
            t (int): Current timestep.

        Returns:
            Estimated mean tensor.
        """
        sqrt_inverse_alpha = self.batch_select_time_indices(self.sqrt_inverse_alphas, t)
        noise_coefficient = self.batch_select_time_indices(self.estimated_mean_noise_coefficient, t)
        estimated_mean = sqrt_inverse_alpha * (x_t - noise_coefficient * predicted_noise)
        return estimated_mean

    @device_grad_decorator(device=None, no_grad=True)
    def estimate_variance(self, t):
        """
        Estimate the variance for the reverse process in DDPM.

        Args:
            t (int): Current timestep.

        Returns:
            Tuple of log variance beta_tilda and log variance beta.
        """
        estimated_log_variance_beta_tilda = self.batch_select_time_indices(self.estimated_log_variance_beta_tilda, t)
        estimated_log_variance_beta = self.batch_select_time_indices(self.estimated_log_variance_beta, t)
        return estimated_log_variance_beta_tilda, estimated_log_variance_beta

    @device_grad_decorator(device=None, no_grad=True)
    def reverse_one_step(self, noisy_mask, t, conditioned_image):
        """
        Sample from the model in DDPM reverse process.

        Args:
            mask (tensor): The input tensor.
            t (int): The current timestep.
            conditioned_image (tensor): Conditioning tensor.

        Returns:
            The predicted previous mask.
        """
        batched_times = torch.full((noisy_mask.shape[0],), t, dtype=torch.long, device=self.device)

        predicted_noise = self.model(noisy_mask, batched_times, conditioned_image)
        model_mean = self.estimate_mean(predicted_noise, noisy_mask, batched_times)
        log_var_beta_tilda, log_var_beta = self.estimate_variance(batched_times)

        noise = torch.randn_like(noisy_mask) if t > 0 else torch.zeros_like(noisy_mask)  # no noise if t == 0
        sigma = (0.5 * log_var_beta).exp()
        predicted_previous_mask = model_mean + sigma * noise
        return predicted_previous_mask

    @device_grad_decorator(device=None, no_grad=True)
    def sample(self, conditioned_image, disable_tqdm=False):
        """
        Generate a sample based on conditioning image using DDPM.

        Args:
            conditioned_image (tensor): Conditioning images tensor with the shape of [batch_size, channels, height, width].
            disable_tqdm (bool): If True, disable the tqdm progress bar.
        Returns:
            Generated masks tensor.
        """
        noisy_mask = torch.randn(
            (conditioned_image.shape[0], self.mask_channels, self.image_size, self.image_size),
            device=self.device,
        )

        for t in tqdm(reversed(range(self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps, disable=disable_tqdm):
            noisy_mask = self.reverse_one_step(noisy_mask, t, conditioned_image)

        denoised_mask = unnormalize_to_zero_to_one(noisy_mask)
        return denoised_mask

    @device_grad_decorator(device=None, no_grad=True)
    def sample_with_snapshots(self, conditioned_image, snapshot_interval: int = None):
        """
        Generator version of sample that yields intermediate noisy masks every snapshot_interval steps in DDPM.

        Args:
            conditioned_image (tensor): Conditioning images.
            snapshot_interval (int): Yield every N steps; if None or > num_timesteps, only yield final.
        Yields:
            tuple(int, tensor): (t, current_noisy_mask) at intervals, and finally the denoised mask.
        """
        if snapshot_interval is None:
            snapshot_interval = self.num_timesteps + 1  # Effectively disable intermediates

        noisy_mask = torch.randn(
            (conditioned_image.shape[0], self.mask_channels, self.image_size, self.image_size),
            device=self.device,
        )

        for t in reversed(range(self.num_timesteps)):
            noisy_mask = self.reverse_one_step(noisy_mask, t, conditioned_image)
            if (self.num_timesteps - t) % snapshot_interval == 0 or t == 0:
                yield t, unnormalize_to_zero_to_one(noisy_mask.clone())  # Yield normalized copy

        final_denoised = unnormalize_to_zero_to_one(noisy_mask)
        yield 0, final_denoised  # Final yield

    @device_grad_decorator(device=None)
    def calculate_xt_from_x0(self, x_0, t, noise):
        """
        Compute noisy image from original in DDPM forward process.

        Args:
            x_0 (tensor): The start image.
            t (int): The current timestep.
            noise (tensor): Optional noise tensor.

        Returns:
            Sampled image tensor.
        """
        first_term = self.batch_select_time_indices(self.sqrt_alpha_bars, t) * x_0
        second_term = self.batch_select_time_indices(self.sqrt_one_minus_alpha_bars, t) * noise
        return first_term + second_term

    @device_grad_decorator(device=None)
    def forward(self, mask, conditioned_image, return_intermediates=False, *args, **kwargs):
        """
        Forward pass through the DDPM model.

        Args:
            mask (tensor): Input mask tensor.
            conditioned_image (tensor): Conditioning image tensor.
            return_intermediates (bool): If True, return intermediates.
            args: Additional arguments.
            kwargs: Keyword arguments.

        Returns:
            The average loss, per_sample_mses, t
        """
        mask, conditioned_image = mask.to(self.device), conditioned_image.to(self.device)
        t = torch.randint(0, self.num_timesteps, [mask.shape[0]], device=self.device).long()

        mask = normalize_to_neg_one_to_one(mask)
        noise = torch.randn(mask.shape, device=self.device)

        x_t = self.calculate_xt_from_x0(mask, t, noise)
        noise_hat = self.model(x_t, t, conditioned_image)

        # Compute per-sample MSE
        sample_mses = torch.mean((noise_hat - noise) ** 2, dim=[1,2,3])
        loss = torch.mean(sample_mses)

        if return_intermediates:
            intermediates = {
                'img': conditioned_image,
                'mask': mask,
                'x_t': x_t,
                'noise': noise,
                'noise_hat': noise_hat
            }
            return loss, sample_mses, t, intermediates
        return loss, sample_mses, t
