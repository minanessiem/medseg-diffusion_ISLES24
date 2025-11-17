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
from ..models.wrappers.conditional_wrapper import ConditionalModelWrapper
from ..models.MedSegDiff.unet_util import (
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
)

# Import auxiliary loss functions
from ..losses.segmentation_losses import DiceLoss, BCELoss


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
        
        # Parse auxiliary losses config from cfg.loss.auxiliary_losses
        # Required config structure (see configs/loss/mse_only.yaml or multitask.yaml):
        # auxiliary_losses:
        #   enabled: bool
        #   diffusion_weight: float
        #   dice: {enabled: bool, weight: float, smooth: float, apply_sigmoid: bool}
        #   bce: {enabled: bool, weight: float, pos_weight: float|null, apply_sigmoid: bool}
        #   warmup_steps: int
        aux_cfg = cfg.loss.auxiliary_losses
        
        if not aux_cfg.enabled:
            # Auxiliary losses explicitly disabled
            self.aux_losses_enabled = False
            self.diffusion_weight = None
            self.dice_loss_fn = None
            self.dice_weight = 0
            self.bce_loss_fn = None
            self.bce_weight = 0
            self.aux_warmup_steps = 0
            return
        
        # Auxiliary losses enabled - parse all parameters
        self.aux_losses_enabled = True
        self.diffusion_weight = aux_cfg.diffusion_weight
        self.aux_warmup_steps = aux_cfg.warmup_steps
        
        # Validate warmup_steps
        if not isinstance(self.aux_warmup_steps, int) or self.aux_warmup_steps < 0:
            self.aux_warmup_steps = 0
        
        # Initialize Dice loss
        if aux_cfg.dice.enabled and aux_cfg.dice.weight > 0:
            self.dice_loss_fn = DiceLoss(
                smooth=aux_cfg.dice.smooth,
                apply_sigmoid=aux_cfg.dice.apply_sigmoid
            )
            self.dice_weight = aux_cfg.dice.weight
        else:
            self.dice_loss_fn = None
            self.dice_weight = 0
        
        # Initialize BCE loss
        if aux_cfg.bce.enabled and aux_cfg.bce.weight > 0:
            self.bce_loss_fn = BCELoss(
                pos_weight=aux_cfg.bce.pos_weight,
                apply_sigmoid=aux_cfg.bce.apply_sigmoid
            )
            self.bce_weight = aux_cfg.bce.weight
        else:
            self.bce_loss_fn = None
            self.bce_weight = 0
        
        # Console output
        print("[OpenAI Adapter] Multi-task loss enabled:")
        print(f"  Diffusion weight: {self.diffusion_weight}")
        print(f"  Dice: enabled={self.dice_loss_fn is not None}, weight={self.dice_weight}")
        print(f"  BCE: enabled={self.bce_loss_fn is not None}, weight={self.bce_weight}")
        print(f"  Warmup steps: {self.aux_warmup_steps}")
    
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
    
    def forward(self, mask, conditioned_image, return_intermediates=False, global_step=0, *args, **kwargs):
        """
        Training forward pass with optional multi-task loss support.
        
        This method computes the diffusion loss for a batch of masks and
        conditioning images. When auxiliary losses are enabled and past warmup,
        it also computes Dice and BCE losses on the reconstructed mask pred_x0.
        
        Args:
            mask (torch.Tensor): Target masks [B, 1, H, W] in [0, 1]
            conditioned_image (torch.Tensor): Conditioning images [B, C, H, W]
            return_intermediates (bool): If True, return intermediate tensors
                                       for image logging (img, mask, x_t, noise, noise_hat, pred_x0)
            global_step (int): Current training step (for warmup logic). Default: 0
            
        Returns:
            If return_intermediates=False:
                loss (torch.Tensor): Scalar loss
                sample_mses (torch.Tensor): Per-sample MSE losses [B]
                t (torch.Tensor): Sampled timesteps [B]
                loss_components (dict or None): Individual loss components if aux losses enabled
                    Keys: 'mse', 'dice' (optional), 'bce' (optional), 'total'
                    None if aux losses disabled
            
            If return_intermediates=True:
                loss, sample_mses, t, intermediates (dict)
                intermediates contains:
                    - 'img': conditioned_image [B, C, H, W]
                    - 'mask': normalized mask [B, 1, H, W] in [-1, 1]
                    - 'x_t': noisy mask [B, 1, H, W]
                    - 'noise': ground truth noise [B, 1, H, W]
                    - 'noise_hat': predicted noise [B, 1, H, W]
                    - 'pred_x0': reconstructed mask from noise prediction [B, 1, H, W] in [-1, 1]
                    - 'loss_components': dict of individual loss values (if multi-task enabled)
                      Keys: 'mse', 'dice' (optional), 'bce' (optional), 'total'
        
        Notes:
            - Masks are automatically normalized from [0, 1] to [-1, 1]
            - When auxiliary losses are enabled and global_step >= warmup_steps,
              total loss = diffusion_weight * MSE + dice_weight * Dice + bce_weight * BCE
            - Auxiliary losses operate on pred_x0 denormalized to [0, 1] range
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
        
        # Compute primary diffusion loss (MSE on noise)
        # Note: This assumes model_mean_type=EPSILON (noise prediction)
        # For START_X or PREVIOUS_X modes, target would be different
        sample_mses = torch.mean((noise_hat - noise) ** 2, dim=[1, 2, 3])
        mse_loss = sample_mses.mean()
        
        # Initialize total loss and components dict
        if self.aux_losses_enabled:
            loss = self.diffusion_weight * mse_loss
            loss_components = {'mse': mse_loss.item()}
        else:
            loss = mse_loss
            loss_components = None
        
        # Compute auxiliary losses if enabled and past warmup
        pred_x0 = None  # Will be computed if needed
        if self.aux_losses_enabled and global_step >= self.aux_warmup_steps:
            # Reconstruct pred_x0 for auxiliary losses
            # Only computed when needed - zero overhead if disabled
            pred_x0 = self.diffusion._predict_xstart_from_eps(x_t=x_t, t=t, eps=noise_hat)
            # Clamp to [-1, 1] range (reconstruction can produce values outside this range)
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
            # Denormalize pred_x0 from [-1, 1] to [0, 1] for loss computation
            # (Dice/BCE expect [0, 1] range, not [-1, 1])
            pred_x0_denorm = unnormalize_to_zero_to_one(pred_x0)
            mask_denorm = mask  # Already in [0, 1]
            
            # Dice loss
            if self.dice_loss_fn is not None and self.dice_weight > 0:
                dice_loss = self.dice_loss_fn(pred_x0_denorm, mask_denorm)
                loss = loss + self.dice_weight * dice_loss
                loss_components['dice'] = dice_loss.item()
            
            # BCE loss
            if self.bce_loss_fn is not None and self.bce_weight > 0:
                bce_loss = self.bce_loss_fn(pred_x0_denorm, mask_denorm)
                loss = loss + self.bce_weight * bce_loss
                loss_components['bce'] = bce_loss.item()
            
            # Add total loss for logging
            loss_components['total'] = loss.item()
        
        # Compute pred_x0 for logging if needed (even if aux losses disabled)
        if return_intermediates:
            if pred_x0 is None:
                # Not yet computed (aux losses disabled or in warmup)
                pred_x0 = self.diffusion._predict_xstart_from_eps(x_t=x_t, t=t, eps=noise_hat)
                # Clamp to [-1, 1] range (reconstruction can produce values outside this range)
                pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
            intermediates = {
                'img': conditioned_image,      # [B, C, H, W] - conditioning
                'mask': mask_normalized,       # [B, 1, H, W] - target in [-1, 1]
                'x_t': x_t,                    # [B, 1, H, W] - noisy mask
                'noise': noise,                # [B, 1, H, W] - ground truth noise
                'noise_hat': noise_hat,        # [B, 1, H, W] - predicted noise
                'pred_x0': pred_x0,            # [B, 1, H, W] - reconstructed mask from noise prediction
            }
            
            # Add loss components for logging
            if loss_components is not None:
                intermediates['loss_components'] = loss_components
            
            return loss, sample_mses, t, intermediates
        
        # Return loss_components even without intermediates for logging
        return loss, sample_mses, t, loss_components
    
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

