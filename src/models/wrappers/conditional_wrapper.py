"""
Conditional model wrapper for OpenAI diffusion compatibility.

This module provides a thin wrapper around our conditional UNet to make it
compatible with OpenAI's diffusion interface expectations.
"""

import torch.nn as nn


class ConditionalModelWrapper(nn.Module):
    """
    Wraps a conditional UNet to work with OpenAI's diffusion interface.
    
    OpenAI's diffusion expects models with signature:
        model(x, t, **model_kwargs)
    
    Our conditional UNet has signature:
        unet(x, t, conditioned_image)
    
    This wrapper extracts the conditioning from model_kwargs and passes it
    to the base UNet as a positional argument.
    
    Args:
        base_model (nn.Module): The conditional UNet (possibly wrapped in DataParallel)
        condition_key (str): Key to extract conditioning tensor from model_kwargs.
                           Default: 'conditioned_image'
    
    Example:
        >>> unet = Unet(cfg)
        >>> wrapped = ConditionalModelWrapper(unet)
        >>> output = wrapped(x_t, t, conditioned_image=img)  # kwargs-based
        >>> # Equivalent to: unet(x_t, t, img)
    
    Notes:
        - The wrapper preserves important properties from base_model (image_channels, etc.)
        - Works correctly when base_model is already wrapped in DataParallel
        - Provides helpful error messages when conditioning is missing
        - Preserves tuple output structure (e.g., ORGMedSegDiff returns (noise_pred, cal))
        - Exposes produces_calibration flag from base model if present
    """
    
    def __init__(self, base_model, condition_key='conditioned_image'):
        super().__init__()
        self.base_model = base_model
        self.condition_key = condition_key
        
        # Preserve properties from base model (may be wrapped in DataParallel)
        self.image_channels = self._get_attr('image_channels')
        self.mask_channels = self._get_attr('mask_channels')
        self.output_channels = self._get_attr('output_channels')
        self.image_size = self._get_attr('image_size')
        
        # Flag for models that return tuple (noise_pred, calibration)
        # e.g., ORGMedSegDiff with highway network
        self.produces_calibration = self._get_attr('produces_calibration') or False
    
    def _get_attr(self, name):
        """Helper to get attribute from potentially wrapped model."""
        if hasattr(self.base_model, name):
            return getattr(self.base_model, name)
        elif hasattr(self.base_model, 'module') and hasattr(self.base_model.module, name):
            # Handle DataParallel wrapping
            return getattr(self.base_model.module, name)
        return None
    
    def forward(self, x, t, **model_kwargs):
        """
        Forward pass that extracts conditioning from kwargs.
        
        Args:
            x (torch.Tensor): Noisy input [B, C, H, W]
            t (torch.Tensor): Timesteps [B] or [B, 1]
            **model_kwargs: Must contain key specified by self.condition_key
            
        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]:
                - Standard models: Predicted noise [B, C, H, W]
                  (or [B, 2*C, H, W] if model predicts variance)
                - ORGMedSegDiff: Tuple of (noise_pred, calibration)
                  where calibration is [B, 1, H, W] with sigmoid applied
        
        Raises:
            ValueError: If conditioning key is missing from model_kwargs
        """
        conditioned_image = model_kwargs.get(self.condition_key)
        if conditioned_image is None:
            available_keys = list(model_kwargs.keys())
            raise ValueError(
                f"Missing required conditioning key '{self.condition_key}' in model_kwargs. "
                f"Available keys: {available_keys}. "
                f"Ensure diffusion.forward() or diffusion.sample() passes "
                f"{self.condition_key}=<conditioning_tensor> in model_kwargs."
            )
        
        return self.base_model(x, t, conditioned_image)

