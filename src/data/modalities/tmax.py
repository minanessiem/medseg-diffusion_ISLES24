import torch
from monai.transforms import ScaleIntensityRange

def process_tmax(data: torch.Tensor, max_val: float, min_val: float,
                ROI_mask: bool = False,
                description: str = None, threshold: float = None, **kwargs) -> torch.Tensor:
    """
    Process TMAX images, returning either normalized data or a mask based on a value range,
    following a structure similar to process_cbf.
    
    Args:
        data: Input tensor of shape [H, W, D] or [1, H, W, D].
        max_val: Maximum value for normalization or upper bound for ROI mask range.
        min_val: Minimum value for normalization or lower bound for ROI mask range.
        ROI_mask: If True, return a binary mask within the [min_val, max_val] range;
                  if False, return normalized TMAX values.
        description: Optional description string.
        threshold: Original threshold parameter (potentially unused if ROI_mask=True,
                   kept for compatibility or future use).
        **kwargs: Additional parameters (ignored).
    
    Returns:
        Tensor of shape [1, H, W, D] containing either:
        - Normalized TMAX values (if ROI_mask=False).
        - Binary mask where min_val < TMAX < max_val (if ROI_mask=True).
    """

    # Step 1: Create a background mask (exclude near-zero values)
    # Similar to brain_mask in process_cbf
    background_mask = (data > 1e-5).float()

    # Step 2: Clip data (using 0 and max_val, similar to process_cbf's clipping approach)
    # Note: Clipping uses 0 as lower bound, mirroring process_cbf logic,
    # even if min_val for normalization/masking is different.
    data_clipped = torch.clamp(data, 0, max_val)

    if ROI_mask:
        # Step 3a: Return the mask for the range (min_val, max_val)
        # Apply range check to clipped data and ensure it's within the background mask
        # Mimics the structure: (condition_on_clipped_data) & (background_mask > effective_lower_bound)
        # Here, the effective lower bound for the mask check is 0
        threshold_mask = (
            (data_clipped > min_val) &
            (data_clipped < max_val) &
            (background_mask > 0) # Equivalent to applying the background mask
        ).float()
        return threshold_mask
    else:
        # Step 3b: Return normalized data using the specified [min_val, max_val] range
        transform = ScaleIntensityRange(
            a_min=min_val,
            a_max=max_val,
            b_min=0,
            b_max=1,
            clip=True
        )
        # Apply transform to the original data, as done in process_cbf
        normalized = transform(data)
        return normalized

def get_tmax_params(config: str, data_stats: dict) -> dict:
    """Get TMAX-specific parameters for processing"""
    params = {
        'TMAX_min_0_max_4': {
            'threshold': None,
            'min_val': 0,
            'max_val': 4,
            'ROI_mask': False,
            'description': 'TMAX normalized: min 0, max 4'
        },
        'TMAX_min_0_max_6': {
            'threshold': None,
            'min_val': 0,
            'max_val': 6,
            'ROI_mask': False,
            'description': 'TMAX normalized: min 0, max 6'
        },
        'TMAX_min_0_max_8': {
            'threshold': None,
            'min_val': 0,
            'max_val': 8,
            'ROI_mask': False,
            'description': 'TMAX normalized: min 0, max 8'
        },
        'TMAX_min_0_max_10': {
            'threshold': None,
            'min_val': 0,
            'max_val': 10,
            'ROI_mask': False,
            'description': 'TMAX normalized: min 0, max 10'
        },
        'TMAX_min_0_max_12': {
            'threshold': None,
            'min_val': 0,
            'max_val': 12,
            'ROI_mask': False,
            'description': 'TMAX normalized: min 0, max 12'
        },
        'TMAX_min_0_max_16': {
            'threshold': None,
            'min_val': 0,
            'max_val': 16,
            'ROI_mask': False,
            'description': 'TMAX normalized: min 0, max 16'
        },
        'TMAX_min_0_max_30': {
            'threshold': None,
            'min_val': 0,
            'max_val': 30,
            'ROI_mask': False,
            'description': 'TMAX normalized: min 0, max 30'
        },
        'TMAX_min_0_max_maxval': {
            'threshold': None,
            'min_val': 0,
            'max_val': data_stats['max_val'],
            'ROI_mask': False,
            'description': 'TMAX normalized: min 0, max MAX_VAL'
        },
        'TMAX_min_4_max_10': {
            'threshold': None,
            'min_val': 4,
            'max_val': 10,
            'ROI_mask': False,
            'description': 'TMAX normalized: min 4, max 10'
        },
        'TMAX_min_6_max_10': {
            'threshold': None,
            'min_val': 6,
            'max_val': 10,
            'ROI_mask': False,
            'description': 'TMAX normalized: min 6, max 10'
        },
        'TMAX_min_4_max_30': {
            'threshold': None,
            'min_val': 4,
            'max_val': 30,
            'ROI_mask': False,
            'description': 'TMAX normalized: min 4, max 30'
        },
        'TMAX_min_5_max_30': {
            'threshold': None,
            'min_val': 5,
            'max_val': 30,
            'ROI_mask': False,
            'description': 'TMAX normalized: min 5, max 30'
        },
        'TMAX_min_6_max_30': {
            'threshold': None,
            'min_val': 6,
            'max_val': 30,
            'ROI_mask': False,
            'description': 'TMAX normalized: min 6, max 30'
        },
        'TMAX_min_8_max_30': {
            'threshold': None,
            'min_val': 8,
            'max_val': 30,
            'ROI_mask': False,
            'description': 'TMAX normalized: min 8, max 30'
        },
        'TMAX_min_10_max_30': {
            'threshold': None,
            'min_val': 10,
            'max_val': 30,
            'ROI_mask': False,
            'description': 'TMAX normalized: min 10, max 30'
        },

        # ROI Mask configurations (range-based)
        'TMAX_mask_min_4_max_maxval': {
            'threshold': None,
            'min_val': 4,
            'max_val': data_stats['max_val'],
            'ROI_mask': True,
            'description': 'TMAX Mask: 4 < TMAX < MAX_VAL'
        },
        'TMAX_mask_min_6_max_maxval': {
            'threshold': None,
            'min_val': 6,
            'max_val': data_stats['max_val'],
            'ROI_mask': True,
            'description': 'TMAX Mask: 6 < TMAX < MAX_VAL'
        },
        'TMAX_mask_min_8_max_maxval': {
            'threshold': None,
            'min_val': 8,
            'max_val': data_stats['max_val'],
            'ROI_mask': True,
            'description': 'TMAX Mask: 8 < TMAX < MAX_VAL'
        },
        'TMAX_mask_min_10_max_maxval': {
            'threshold': None,
            'min_val': 10,
            'max_val': data_stats['max_val'],
            'ROI_mask': True,
            'description': 'TMAX Mask: 10 < TMAX < MAX_VAL'
        },
        'TMAX_mask_min_16_max_maxval': {
            'threshold': None,
            'min_val': 16,
            'max_val': data_stats['max_val'],
            'ROI_mask': True,
            'description': 'TMAX Mask: 16 < TMAX < MAX_VAL'
        },
    }
    
    if config not in params:
        raise ValueError(f"Unknown TMAX configuration: {config}")
        
    return params[config] 
