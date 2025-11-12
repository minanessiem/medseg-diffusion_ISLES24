"""
OpenAI GaussianDiffusion adapter for medical image segmentation.

This module provides an adapter that bridges OpenAI's improved_diffusion
package to our existing Diffusion interface, enabling use of their
battle-tested DDPM/DDIM implementations while maintaining compatibility
with our training loop, logger, and conditional UNet.

Key features:
- Full DDPM and DDIM support (configurable)
- Complete intermediates for image logging
- Multi-GPU compatibility (DataParallel)
- Spaced diffusion for faster sampling
- Loss-aware sampling support
"""

import torch
import torch.nn as nn
from omegaconf import OmegaConf
import warnings

from .diffusion import Diffusion
from ..improved_diffusion import (
    GaussianDiffusion,
    SpacedDiffusion,
    space_timesteps,
    get_named_beta_schedule,
    ModelMeanType,
    ModelVarType,
    LossType,
    UniformSampler,
    LossSecondMomentResampler,
)
from ..models.architectures.conditional_wrapper import ConditionalModelWrapper
from ..models.architectures.unet_util import (
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
)


class GaussianDiffusionAdapter(Diffusion):
    """
    Adapter wrapping OpenAI's GaussianDiffusion for our medical segmentation pipeline.
    
    This adapter implements our Diffusion interface while delegating the core
    diffusion mathematics to OpenAI's implementation. It handles:
    
    1. Configuration mapping (Hydra -> OpenAI parameters)
    2. Model wrapping (conditional UNet -> OpenAI interface)
    3. Intermediates computation (for image logging)
    4. DDPM/DDIM mode switching
    5. Multi-GPU compatibility
    
    The adapter keeps OpenAI's code pristine while providing seamless
    integration with our existing trainer, logger, and data pipeline.
    
    Args:
        model (nn.Module): Conditional UNet (possibly wrapped in DataParallel)
        cfg (DictConfig): Hydra configuration object
        device (torch.device, optional): Target device
    
    Example:
        >>> unet = Unet(cfg)
        >>> adapter = GaussianDiffusionAdapter(unet, cfg, device)
        >>> loss, mses, t = adapter.forward(mask, img)
        >>> samples = adapter.sample(img)  # DDPM or DDIM based on config
    
    Config Options:
        See configs/diffusion/openai_base.yaml for full parameter documentation.
    """
    
    def __init__(self, model, cfg, device=None):
        super().__init__(model, cfg, device)
        
        OmegaConf.set_struct(cfg, False)
        
        # Parse core diffusion parameters
        self.timesteps = cfg.diffusion.timesteps
        self.noise_schedule = cfg.diffusion.noise_schedule
        self.model_mean_type_str = cfg.diffusion.model_mean_type
        self.model_var_type_str = cfg.diffusion.model_var_type
        self.loss_type_str = cfg.diffusion.loss_type
        self.rescale_timesteps = cfg.diffusion.get('rescale_timesteps', False)
        self.timestep_respacing = cfg.diffusion.get('timestep_respacing', '')
        
        # Sampling parameters
        self.sampling_mode = cfg.diffusion.get('sampling_mode', 'ddpm')
        self.ddim_eta = cfg.diffusion.get('ddim_eta', 0.0)
        
        # Loss-aware sampling (note: requires DDP, not compatible with DataParallel)
        self.use_loss_aware_sampling = cfg.diffusion.get('use_loss_aware_sampling', False)
        
        OmegaConf.set_struct(cfg, True)
        
        # Validate configuration
        self._validate_config()
        
        # Wrap model for OpenAI compatibility
        self.wrapped_model = ConditionalModelWrapper(model, condition_key='conditioned_image')
        
        # Create OpenAI diffusion instance
        self.diffusion = self._create_diffusion()
        
        # Expose num_timesteps for logger compatibility
        self.num_timesteps = self.diffusion.num_timesteps
        
        # Note: Schedule sampler not currently used in v1.0 (we use uniform random sampling in forward())
        # Kept for future integration with OpenAI's training loop
        if self.use_loss_aware_sampling:
            # Use concrete implementation (not abstract LossAwareSampler)
            self.schedule_sampler = LossSecondMomentResampler(self.diffusion)
        else:
            self.schedule_sampler = UniformSampler(self.diffusion)
    
    def _validate_config(self):
        """
        Validate configuration and warn about potential issues.
        
        Checks:
        - Learned variance requires doubled output channels
        - Large timesteps benefit from rescale_timesteps
        - DDIM eta in valid range
        - Loss-aware sampling compatibility with distributed training
        - Model mean type compatibility with manual loss computation
        """
        # Check learned variance mode
        if self.model_var_type_str in ['LEARNED', 'LEARNED_RANGE']:
            expected_output = self.mask_channels * 2
            actual_output = self.model.output_channels
            if actual_output != expected_output:
                raise ValueError(
                    f"Configuration error: model_var_type={self.model_var_type_str} "
                    f"requires model.output_channels={expected_output}, but model has "
                    f"{actual_output}. Set model.output_channels={expected_output} in config "
                    f"or change model_var_type to FIXED_SMALL/FIXED_LARGE."
                )
        
        # Warn about time embedding scale
        if self.timesteps >= 500 and not self.rescale_timesteps:
            warnings.warn(
                f"Using {self.timesteps} timesteps without rescale_timesteps=true. "
                f"This may cause time embedding saturation. Consider setting "
                f"diffusion.rescale_timesteps=true in your config for better results.",
                UserWarning
            )
        
        # Validate DDIM eta
        if not (0.0 <= self.ddim_eta <= 1.0):
            raise ValueError(
                f"DDIM eta must be in [0, 1], got {self.ddim_eta}. "
                f"Use 0.0 for deterministic, 1.0 for stochastic DDIM."
            )
        
        # Warn about loss-aware sampling with DataParallel
        if self.use_loss_aware_sampling:
            warnings.warn(
                "Loss-aware sampling requires torch.distributed (DDP) but you may be using "
                "DataParallel. This feature is currently non-functional in v1.0 and will be "
                "ignored during training. To use loss-aware sampling, migrate to DDP.",
                UserWarning
            )
        
        # Validate model_mean_type for manual loss computation
        if self.model_mean_type_str != 'EPSILON':
            warnings.warn(
                f"model_mean_type={self.model_mean_type_str} detected. The adapter's manual "
                f"loss computation assumes EPSILON mode (predicting noise). Other modes may "
                f"produce incorrect training loss. Recommended: use model_mean_type=EPSILON.",
                UserWarning
            )
    
    def _create_diffusion(self):
        """
        Create OpenAI GaussianDiffusion or SpacedDiffusion instance.
        
        Returns:
            GaussianDiffusion or SpacedDiffusion instance
        """
        # Get beta schedule
        betas = get_named_beta_schedule(self.noise_schedule, self.timesteps)
        
        # Map string config to OpenAI enums
        model_mean_type = getattr(ModelMeanType, self.model_mean_type_str)
        model_var_type = getattr(ModelVarType, self.model_var_type_str)
        loss_type = getattr(LossType, self.loss_type_str)
        
        # Create appropriate diffusion type
        if self.timestep_respacing:
            # Spaced diffusion (fewer sampling steps)
            use_timesteps = space_timesteps(self.timesteps, self.timestep_respacing)
            return SpacedDiffusion(
                use_timesteps=use_timesteps,
                betas=betas,
                model_mean_type=model_mean_type,
                model_var_type=model_var_type,
                loss_type=loss_type,
                rescale_timesteps=self.rescale_timesteps,
            )
        else:
            # Standard DDPM
            return GaussianDiffusion(
                betas=betas,
                model_mean_type=model_mean_type,
                model_var_type=model_var_type,
                loss_type=loss_type,
                rescale_timesteps=self.rescale_timesteps,
            )
    
    def _scale_timesteps(self, t):
        """
        Scale timesteps if rescale_timesteps is enabled.
        
        Args:
            t (torch.Tensor): Raw timestep indices
            
        Returns:
            torch.Tensor: Scaled timesteps (if enabled) or original
        """
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.diffusion.num_timesteps)
        return t
    
    def forward(self, mask, conditioned_image, return_intermediates=False, *args, **kwargs):
        """
        Training forward pass with full intermediates support.
        
        This method computes the diffusion loss for a batch of masks and
        conditioning images. When return_intermediates=True, it also captures
        all intermediate tensors needed for image logging in the trainer.
        
        Args:
            mask (torch.Tensor): Target masks [B, 1, H, W] in [0, 1]
            conditioned_image (torch.Tensor): Conditioning images [B, C, H, W]
            return_intermediates (bool): If True, return intermediate tensors
                                       for image logging (img, mask, x_t, noise, noise_hat)
            
        Returns:
            If return_intermediates=False:
                loss (torch.Tensor): Scalar loss
                sample_mses (torch.Tensor): Per-sample MSE losses [B]
                t (torch.Tensor): Sampled timesteps [B]
            
            If return_intermediates=True:
                loss, sample_mses, t, intermediates (dict)
                intermediates contains:
                    - 'img': conditioned_image [B, C, H, W]
                    - 'mask': normalized mask [B, 1, H, W] in [-1, 1]
                    - 'x_t': noisy mask [B, 1, H, W]
                    - 'noise': ground truth noise [B, 1, H, W]
                    - 'noise_hat': predicted noise [B, 1, H, W]
        
        Notes:
            - Masks are automatically normalized from [0, 1] to [-1, 1]
            - This implementation manually computes x_t and noise_hat to
              capture intermediates, then computes loss directly rather than
              calling OpenAI's training_losses() (which doesn't expose intermediates)
        """
        mask = mask.to(self.device)
        conditioned_image = conditioned_image.to(self.device)
        batch_size = mask.shape[0]
        
        # Normalize mask to [-1, 1] (OpenAI convention)
        mask_normalized = normalize_to_neg_one_to_one(mask)
        
        # Sample random timesteps
        t = torch.randint(0, self.diffusion.num_timesteps, (batch_size,), device=self.device).long()
        
        # Sample noise (explicit for intermediates)
        noise = torch.randn_like(mask_normalized)
        
        # Compute x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        x_t = self.diffusion.q_sample(mask_normalized, t, noise=noise)
        
        # Prepare model kwargs and predict noise
        model_kwargs = {'conditioned_image': conditioned_image}
        noise_hat = self.wrapped_model(x_t, self._scale_timesteps(t), **model_kwargs)
        
        # Compute loss (MSE between predicted and true noise)
        # Note: This assumes model_mean_type=EPSILON (noise prediction)
        # For START_X or PREVIOUS_X modes, target would be different
        sample_mses = torch.mean((noise_hat - noise) ** 2, dim=[1, 2, 3])
        loss = sample_mses.mean()
        
        if return_intermediates:
            intermediates = {
                'img': conditioned_image,      # [B, C, H, W] - conditioning
                'mask': mask_normalized,       # [B, 1, H, W] - target in [-1, 1]
                'x_t': x_t,                    # [B, 1, H, W] - noisy mask
                'noise': noise,                # [B, 1, H, W] - ground truth noise
                'noise_hat': noise_hat,        # [B, 1, H, W] - predicted noise
            }
            return loss, sample_mses, t, intermediates
        
        return loss, sample_mses, t
    
    def sample(self, conditioned_image, disable_tqdm=False):
        """
        Generate samples using DDPM or DDIM based on config.
        
        Sampling mode is determined by self.sampling_mode ('ddpm' or 'ddim').
        DDIM sampling is significantly faster but may have slightly lower quality.
        
        Args:
            conditioned_image (torch.Tensor): Conditioning images [B, C, H, W]
            disable_tqdm (bool): If True, disable progress bar
            
        Returns:
            torch.Tensor: Generated masks [B, 1, H, W] in [0, 1]
        
        Performance:
            - DDPM: ~1000 steps (slower, higher quality)
            - DDIM (50 steps): ~20x faster, comparable quality
            - DDIM (250 steps): ~4x faster, near-identical quality
        """
        conditioned_image = conditioned_image.to(self.device)
        batch_size = conditioned_image.shape[0]
        shape = (batch_size, self.mask_channels, self.image_size, self.image_size)
        model_kwargs = {'conditioned_image': conditioned_image}
        
        with torch.no_grad():
            if self.sampling_mode == 'ddim':
                # DDIM sampling (fast)
                samples = self.diffusion.ddim_sample_loop(
                    model=self.wrapped_model,
                    shape=shape,
                    noise=None,
                    clip_denoised=True,
                    model_kwargs=model_kwargs,
                    device=self.device,
                    progress=not disable_tqdm,
                    eta=self.ddim_eta,
                )
            else:
                # DDPM sampling (default)
                samples = self.diffusion.p_sample_loop(
                    model=self.wrapped_model,
                    shape=shape,
                    noise=None,
                    clip_denoised=True,
                    model_kwargs=model_kwargs,
                    device=self.device,
                    progress=not disable_tqdm,
                )
        
        # Denormalize from [-1, 1] to [0, 1]
        samples = unnormalize_to_zero_to_one(samples)
        return samples
    
    def sample_with_snapshots(self, conditioned_image, snapshot_interval: int = None):
        """
        Sample with intermediate snapshots.
        
        Yields intermediate denoised masks at regular intervals during the
        reverse diffusion process. Useful for visualization and debugging.
        
        Args:
            conditioned_image (torch.Tensor): Conditioning images [B, C, H, W]
            snapshot_interval (int, optional): Yield every N steps. If None,
                                              only yields final result.
            
        Yields:
            tuple: (t, mask) where t is timestep and mask is current denoised
                   mask [B, 1, H, W] in [0, 1]
        
        Notes:
            - Always uses DDPM progressive sampling (p_sample_loop_progressive)
            - If sampling_mode='ddim', this method still uses DDPM for snapshots
              because OpenAI's implementation doesn't provide ddim_sample_loop_progressive
            - Snapshots are denormalized to [0, 1] for direct visualization
        """
        conditioned_image = conditioned_image.to(self.device)
        batch_size = conditioned_image.shape[0]
        
        if snapshot_interval is None:
            snapshot_interval = self.diffusion.num_timesteps + 1  # Only final
        
        shape = (batch_size, self.mask_channels, self.image_size, self.image_size)
        model_kwargs = {'conditioned_image': conditioned_image}
        
        with torch.no_grad():
            step_count = 0
            for sample_dict in self.diffusion.p_sample_loop_progressive(
                model=self.wrapped_model,
                shape=shape,
                noise=None,
                clip_denoised=True,
                model_kwargs=model_kwargs,
                device=self.device,
                progress=False,
            ):
                step_count += 1
                current_sample = sample_dict['sample']
                t = self.diffusion.num_timesteps - step_count
                
                # Yield at intervals or final step
                if step_count % snapshot_interval == 0 or t == 0:
                    denormalized = unnormalize_to_zero_to_one(current_sample.clone())
                    yield t, denormalized

