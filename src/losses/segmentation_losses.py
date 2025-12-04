"""
Differentiable segmentation loss functions for multi-task learning.

These losses are designed to be mathematically identical to their corresponding
metric counterparts in metrics.py, but differentiable for use in training.

Key Design Principles:
1. Mathematical Alignment: Loss formulas match evaluation metrics exactly
2. Soft Predictions: Operate on continuous [0,1] values (no binarization)
3. Explicit Configuration: All parameters passed explicitly (no defaults)
4. Gradient Flow: All operations differentiable for backpropagation

References:
- DiceNativeCoefficient (metrics.py, lines 59-109): Reference implementation
- DiffSwinTr (MICCAI 2023): Multi-task loss approach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


class DiceLoss(nn.Module):
    """
    Differentiable Dice loss for binary segmentation.
    
    This implementation maintains the exact same mathematical formula as
    DiceNativeCoefficient in metrics.py (lines 59-109), but operates on soft 
    predictions without binarization to enable gradient flow.
    
    Mathematical Formula:
        Dice Coefficient: DC = (2 * intersection) / (pred_sum + true_sum + smooth)
        Dice Loss: 1 - DC (for minimization)
    
    Where:
        intersection = sum(y_pred * y_true)  # Element-wise product
        pred_sum = sum(y_pred)
        true_sum = sum(y_true)
    
    Difference from Metric:
        - Metric: Uses binarized predictions (y_pred > 0.5)
        - Loss: Uses soft predictions directly (no threshold)
        - Both use identical formula after binarization/softmax
    
    Gradient Flow:
        ∂L/∂y_pred flows through multiplication and division operations,
        enabling the model to learn segmentation boundaries.
    
    Args:
        smooth (float): Smoothing constant for numerical stability.
                       Matches epsilon in DiceNativeCoefficient.
                       Typical values: 1e-8 (tight) to 1.0 (relaxed).
        apply_sigmoid (bool): Whether to apply sigmoid to predictions.
                             Set to False if model already outputs [0,1].
                             Set to True if model outputs logits.
    
    Example:
        >>> # For pred_x0 already in [0, 1] range
        >>> dice_loss = DiceLoss(smooth=1e-8, apply_sigmoid=False)
        >>> pred_x0 = model_prediction  # [B, 1, H, W] in [0, 1]
        >>> target = ground_truth       # [B, 1, H, W] in [0, 1]
        >>> loss = dice_loss(pred_x0, target)
        >>> loss.backward()  # Gradients flow to model!
        
        >>> # For model outputting logits
        >>> dice_loss = DiceLoss(smooth=1e-8, apply_sigmoid=True)
        >>> logits = model(x)  # [B, 1, H, W] unbounded
        >>> loss = dice_loss(logits, target)
    
    Notes:
        - Operates on entire batch (not per-sample). For per-sample computation,
          use DiceLossPerSample.
        - Empty masks (all zeros) handled by smoothing parameter.
        - Loss value range: [0, 1] where 0 = perfect, 1 = total mismatch.
    """
    
    def __init__(self, smooth, apply_sigmoid):
        super().__init__()
        self.smooth = smooth
        self.apply_sigmoid = apply_sigmoid
    
    def forward(self, y_pred, y_true):
        """
        Compute differentiable Dice loss.
        
        Args:
            y_pred (torch.Tensor): Predictions in [0, 1], shape [B, C, H, W]
                                  (or logits if apply_sigmoid=True)
            y_true (torch.Tensor): Ground truth in [0, 1], same shape as y_pred
        
        Returns:
            torch.Tensor: Scalar Dice loss (1 - Dice coefficient)
        """
        # Ensure same device
        y_pred = y_pred.to(y_true.device)
        
        # Optional sigmoid (if model outputs logits)
        if self.apply_sigmoid:
            y_pred = torch.sigmoid(y_pred)
        
        # Compute sums (same formula as DiceNativeCoefficient)
        pred_sum = torch.sum(y_pred)
        true_sum = torch.sum(y_true)
        
        # Intersection (element-wise product for soft Dice)
        intersection = torch.sum(y_pred * y_true)
        
        # Dice coefficient (identical formula to metric)
        dice_coef = (2.0 * intersection) / (pred_sum + true_sum + self.smooth)
        
        # Return Dice loss (1 - coefficient for minimization)
        dice_loss = 1.0 - dice_coef
        
        return dice_loss


class DiceLossPerSample(nn.Module):
    """
    Differentiable Dice loss computed per sample, then averaged.
    
    This version computes Dice for each sample independently before averaging,
    which can be more stable for batches with varying mask sizes or when
    different samples have very different foreground ratios.
    
    Mathematical Formula (per sample i):
        DC_i = (2 * intersection_i) / (pred_sum_i + true_sum_i + smooth)
        Loss = mean(1 - DC_i)
    
    Advantages over batch-level DiceLoss:
        - More robust to class imbalance across samples
        - Better gradient distribution for heterogeneous batches
        - Prevents large masks from dominating the loss
    
    Disadvantages:
        - Slightly slower than batch-level computation
        - May be less stable for very small masks
    
    Args:
        smooth (float): Smoothing constant for numerical stability
        apply_sigmoid (bool): Whether to apply sigmoid to predictions
    
    Example:
        >>> dice_loss = DiceLossPerSample(smooth=1e-8, apply_sigmoid=False)
        >>> pred = torch.rand(4, 1, 64, 64)  # Batch of 4 samples
        >>> target = torch.randint(0, 2, (4, 1, 64, 64)).float()
        >>> loss = dice_loss(pred, target)
        >>> # Loss is average of 4 per-sample Dice losses
    """
    
    def __init__(self, smooth, apply_sigmoid):
        super().__init__()
        self.smooth = smooth
        self.apply_sigmoid = apply_sigmoid
    
    def forward(self, y_pred, y_true):
        """
        Compute Dice loss per sample, then average across batch.
        
        Args:
            y_pred (torch.Tensor): Predictions, shape [B, C, H, W]
            y_true (torch.Tensor): Ground truth, shape [B, C, H, W]
        
        Returns:
            torch.Tensor: Scalar Dice loss averaged across batch
        """
        y_pred = y_pred.to(y_true.device)
        
        if self.apply_sigmoid:
            y_pred = torch.sigmoid(y_pred)
        
        batch_size = y_pred.shape[0]
        
        # Flatten spatial dimensions but keep batch separate
        y_pred_flat = y_pred.view(batch_size, -1)  # [B, C*H*W]
        y_true_flat = y_true.view(batch_size, -1)
        
        # Compute per-sample Dice
        intersection = torch.sum(y_pred_flat * y_true_flat, dim=1)  # [B]
        pred_sum = torch.sum(y_pred_flat, dim=1)  # [B]
        true_sum = torch.sum(y_true_flat, dim=1)  # [B]
        
        dice_coef = (2.0 * intersection) / (pred_sum + true_sum + self.smooth)  # [B]
        
        # Average Dice loss across batch
        dice_loss = 1.0 - dice_coef.mean()
        
        return dice_loss


class BCELoss(nn.Module):
    """
    Binary Cross Entropy loss wrapper with optional class balancing.
    
    BCE is a pixel-wise loss that complements Dice loss by providing
    stronger gradients for individual pixels. While Dice optimizes global
    overlap, BCE focuses on per-pixel classification accuracy.
    
    Mathematical Formula:
        BCE = -mean(w * y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
    
    Where w is the positive class weight (pos_weight).
    
    Use Cases:
        - Imbalanced datasets: Set pos_weight > 1 to emphasize foreground
        - Boundary refinement: BCE provides sharper gradients at edges
        - Multi-task with Dice: Combines global (Dice) + local (BCE) supervision
    
    Args:
        pos_weight (float or None): Weight for positive class to handle imbalance.
                                   Values > 1 increase recall (find more positives).
                                   Values < 1 increase precision (reduce false positives).
                                   None = no weighting (balanced classes).
                                   Example: For 10% foreground, use pos_weight=9.0
        apply_sigmoid (bool): Whether to apply sigmoid to predictions.
                             Set to False if model already outputs [0,1].
                             Set to True if model outputs logits.
    
    Example:
        >>> # Balanced dataset
        >>> bce_loss = BCELoss(pos_weight=None, apply_sigmoid=False)
        >>> loss = bce_loss(pred_x0, target)
        
        >>> # Highly imbalanced (10% foreground)
        >>> bce_loss = BCELoss(pos_weight=9.0, apply_sigmoid=False)
        >>> loss = bce_loss(pred_x0, target)
        >>> # Foreground pixels weighted 9x more than background
    
    Notes:
        - Requires predictions in (0, 1) range (not [0, 1] - exact 0/1 cause NaN)
        - Typically combined with Dice: total_loss = w_dice * Dice + w_bce * BCE
        - More sensitive to outliers than Dice loss
    """
    
    def __init__(self, pos_weight, apply_sigmoid):
        super().__init__()
        self.pos_weight = pos_weight
        self.apply_sigmoid = apply_sigmoid
    
    def forward(self, y_pred, y_true):
        """
        Compute Binary Cross Entropy loss.
        
        Args:
            y_pred (torch.Tensor): Predictions in [0, 1] (or logits if apply_sigmoid=True)
            y_true (torch.Tensor): Ground truth in [0, 1], same shape as y_pred
        
        Returns:
            torch.Tensor: Scalar BCE loss
        """
        y_pred = y_pred.to(y_true.device)
        
        # Handle pos_weight tensor
        pos_weight_tensor = None
        if self.pos_weight is not None:
            pos_weight_tensor = torch.tensor([self.pos_weight], 
                                            device=y_pred.device, 
                                            dtype=torch.float32)
        
        if self.apply_sigmoid:
            # Use BCEWithLogitsLoss - more numerically stable and AMP-safe
            return F.binary_cross_entropy_with_logits(
                y_pred, y_true, 
                pos_weight=pos_weight_tensor,
                reduction='mean'
            )
        else:
            # Disable autocast for BCE - binary_cross_entropy is unsafe under AMP
            # See: https://pytorch.org/docs/stable/amp.html#ops-that-can-autocast-to-float32
            with autocast(device_type='cuda', enabled=False):
                return F.binary_cross_entropy(
                    y_pred.float(), y_true.float(), 
                    weight=pos_weight_tensor,
                    reduction='mean'
                )


class CalibrationLoss(nn.Module):
    """
    Calibration loss for highway network auxiliary output.
    
    The MedSegDiff highway network produces a direct segmentation prediction
    ('cal' output) with sigmoid already applied internally (via sigmoid_helper).
    This loss supervises that auxiliary prediction using Binary Cross Entropy.
    
    Mathematical Formula:
        L_cal = BCE(cal, mask)
    
    The official MedSegDiff uses weight=10.0 for this loss, making it a strong
    auxiliary signal that helps the highway network learn useful features
    for modulating the main UNet.
    
    Design Rationale:
        - Highway network acts as a conditioning branch for the main diffusion UNet
        - Calibration loss ensures the highway network learns meaningful features
        - Strong supervision (high weight) encourages direct segmentation ability
        - This auxiliary task improves the quality of conditioning features
    
    Args:
        apply_sigmoid (bool): Whether cal output needs sigmoid activation.
                             Default should be False for official MedSegDiff
                             (sigmoid already applied by model via sigmoid_helper).
                             Set to True only if using a modified model that
                             outputs raw logits.
    
    Example:
        >>> # Standard usage with official MedSegDiff (sigmoid already applied)
        >>> cal_loss = CalibrationLoss(apply_sigmoid=False)
        >>> loss = cal_loss(cal, mask)  # Both in [0, 1]
        
        >>> # With raw logits (custom model)
        >>> cal_loss = CalibrationLoss(apply_sigmoid=True)
        >>> loss = cal_loss(logits, mask)
    
    Notes:
        - Only computed when model returns calibration output (tuple)
        - Gracefully skipped for architectures without highway network
        - Official weight: 10.0 (high relative to diffusion MSE ~0.01-0.1)
        - Cal output shape: [B, 1, H, W] matching mask shape
    """
    
    def __init__(self, apply_sigmoid):
        super().__init__()
        self.apply_sigmoid = apply_sigmoid
    
    def forward(self, cal, target):
        """
        Compute calibration loss (BCE between highway output and ground truth).
        
        Args:
            cal (torch.Tensor): Calibration output from highway network [B, 1, H, W]
                               Already in [0, 1] if apply_sigmoid=False
            target (torch.Tensor): Ground truth mask [B, 1, H, W] in [0, 1]
        
        Returns:
            torch.Tensor: Scalar BCE loss
        """
        cal = cal.to(target.device)
        
        if self.apply_sigmoid:
            # Use BCEWithLogits for numerical stability with raw logits
            # BCEWithLogitsLoss is AMP-safe
            return F.binary_cross_entropy_with_logits(cal, target, reduction='mean')
        else:
            # Cal already has sigmoid applied - use regular BCE
            # Clamp to avoid log(0) numerical issues
            cal_clamped = torch.clamp(cal, min=1e-7, max=1 - 1e-7)
            # Disable autocast for BCE - binary_cross_entropy is unsafe under AMP
            # See: https://pytorch.org/docs/stable/amp.html#ops-that-can-autocast-to-float32
            with autocast(device_type='cuda', enabled=False):
                return F.binary_cross_entropy(
                    cal_clamped.float(), target.float(), reduction='mean'
                )


class CombinedSegmentationLoss(nn.Module):
    """
    Combined loss for multi-task learning with configurable weighting.
    
    This convenience wrapper combines Dice and BCE losses with independent
    weights. Useful for quick experimentation, but production code should
    use individual losses in openai_adapter.py for better logging granularity.
    
    Mathematical Formula:
        L_total = w_dice * L_dice + w_bce * L_bce
    
    Where:
        L_dice = 1 - (2 * intersection) / (pred_sum + true_sum + smooth)
        L_bce = -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
    
    Design Rationale:
        - Dice: Global overlap metric, good for segmentation quality
        - BCE: Pixel-wise accuracy, good for boundary refinement
        - Combined: Leverages strengths of both approaches
    
    Args:
        dice_weight (float): Weight for Dice loss component
        bce_weight (float): Weight for BCE loss component
        dice_smooth (float): Smoothing parameter for Dice loss
        bce_pos_weight (float or None): Positive class weight for BCE
    
    Example:
        >>> # Equal weighting
        >>> combined = CombinedSegmentationLoss(
        ...     dice_weight=0.5, 
        ...     bce_weight=0.5,
        ...     dice_smooth=1e-8, 
        ...     bce_pos_weight=None
        ... )
        >>> loss = combined(pred_x0, target)
        
        >>> # Emphasize Dice (global overlap)
        >>> combined = CombinedSegmentationLoss(
        ...     dice_weight=0.7, 
        ...     bce_weight=0.3,
        ...     dice_smooth=1e-8, 
        ...     bce_pos_weight=None
        ... )
    
    Notes:
        - Set weight=0.0 to disable a component without removing it
        - Weights don't need to sum to 1.0 (can use absolute scales)
        - For production, prefer separate losses in adapter for better logging
    """
    
    def __init__(self, dice_weight, bce_weight, dice_smooth, bce_pos_weight):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        
        # Only instantiate losses if they will be used
        self.dice_loss = DiceLoss(smooth=dice_smooth, apply_sigmoid=False) if dice_weight > 0 else None
        self.bce_loss = BCELoss(pos_weight=bce_pos_weight, apply_sigmoid=False) if bce_weight > 0 else None
    
    def forward(self, y_pred, y_true):
        """
        Compute weighted combination of losses.
        
        Args:
            y_pred (torch.Tensor): Predictions in [0, 1]
            y_true (torch.Tensor): Ground truth in [0, 1]
        
        Returns:
            torch.Tensor: Weighted sum of losses
        """
        total_loss = 0.0
        
        if self.dice_loss is not None and self.dice_weight > 0:
            total_loss += self.dice_weight * self.dice_loss(y_pred, y_true)
        
        if self.bce_loss is not None and self.bce_weight > 0:
            total_loss += self.bce_weight * self.bce_loss(y_pred, y_true)
        
        return total_loss

