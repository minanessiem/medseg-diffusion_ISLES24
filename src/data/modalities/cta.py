import torch
from monai.transforms import ScaleIntensityRange


def process_cta(data: torch.Tensor, window: float = None, level: float = None,
                use_direct_scaling: bool = False, min_val: float = None, max_val: float = None,
                description: str = None, **kwargs) -> torch.Tensor:
    """
    Process CTA images with window/level normalization or direct min/max scaling
    
    Args:
        data: Input tensor
        window: Window width for standard window/level normalization
        level: Window level for standard window/level normalization
        use_direct_scaling: If True, use min_val and max_val directly instead of window/level
        min_val: Minimum value for direct scaling (used only if use_direct_scaling=True)
        max_val: Maximum value for direct scaling (used only if use_direct_scaling=True)
        description: Optional description string
        **kwargs: Additional parameters (ignored)
    
    Returns:
        Tensor of shape [1, H, W, D] containing normalized CTA values
    """

    if use_direct_scaling:
        # Use direct min/max scaling when flag is set
        a_min = min_val
        a_max = max_val
    else:
        # Use traditional window/level calculation
        a_min = level - window/2
        a_max = level + window/2
        
    transform = ScaleIntensityRange(
        a_min=a_min,
        a_max=a_max,
        b_min=0,
        b_max=1,
        clip=True
    )
    normalized = transform(data)
    return normalized

def get_cta_params(config: str, data_stats: dict) -> dict:
    """Get CTA-specific parameters for processing"""
    params = {
        # Add new configuration for direct min/max scaling
        'CTA_min_minval_max_maxval': {
            'use_direct_scaling': True,
            'min_val': data_stats['min_val'],
            'max_val': data_stats['max_val'],
            'description': 'CTA with direct min to max scaling (full range)'
        },
        'CTA_standard_w_80_l_40': {
            'window': 80,
            'level': 40,
            'description': 'CTA Standard, w: 80, l: 40'
        },
        # Impact of Window Setting Optimization on Accuracy of Computed Tomography and Computed Tomography Angiography Source Image-based Alberta Stroke Program Early Computed Tomography Score, Arsava et al.
        # own
        'CTA_w_29_l_35': {
                'window': 29,
                'level': 35,
                'description': 'CTA, w: 29, l: 35'
        },
        'CTA_w_29_l_41': {
            'window': 29,
            'level': 41,
            'description': 'CTA, w: 29, l: 41'
        },
        'CTA_w_29_l_45': {
                'window': 29,
                'level': 45,
                'description': 'CTA, w: 29, l: 45'
        },
        'CTA_w_29_l_49': {
                'window': 29,
                'level': 49,
                'description': 'CTA, w: 29, l: 49'
        },
        # own
        'CTA_w_37_l_35': {
                'window': 37,
                'level': 35,
                'description': 'CTA, w: 37, l: 35'
        },
        'CTA_w_37_l_41': {
                'window': 37,
                'level': 41,
                'description': 'CTA, w: 37, l: 41'
        },
        'CTA_w_37_l_45': {
                'window': 37,
                'level': 45,
                'description': 'CTA, w: 37, l: 45'
        },
        'CTA_w_37_l_49': {
                'window': 37,
                'level': 49,
                'description': 'CTA, w: 37, l: 49'
        },
        # own
        'CTA_w_45_l_35': {
                'window': 45,
                'level': 35,
                'description': 'CTA, w: 45, l: 35'
        },
        'CTA_w_45_l_41': {
                'window': 45,
                'level': 41,
                'description': 'CTA, w: 45, l: 41'
        },
        'CTA_w_45_l_45': {
                'window': 45,
                'level': 45,
                'description': 'CTA, w: 45, l: 45'
        },
        'CTA_w_45_l_49': {
                'window': 45,
                'level': 49,
                'description': 'CTA, w: 45, l: 49'
        },
        # own
        'CTA_w_53_l_35': {
                'window': 53,
                'level': 35,
                'description': 'CTA, w: 53, l: 35'
        },
        'CTA_w_53_l_41': {
                'window': 53,
                'level': 41,
                'description': 'CTA, w: 53, l: 41'
        },
        'CTA_w_53_l_45': {
                'window': 53,
                'level': 45,
                'description': 'CTA, w: 53, l: 45'
        },
        'CTA_w_53_l_49': {
                'window': 53,
                'level': 49,
                'description': 'CTA, w: 53, l: 49'
        },
        # own
        'CTA_w_57_l_35': {
                'window': 57,
                'level': 35,
                'description': 'CTA, w: 57, l: 35'
        },
        'CTA_w_57_l_41': {
                'window': 57,
                'level': 41,
                'description': 'CTA, w: 57, l: 41'
        },
        'CTA_w_57_l_45': {
                'window': 57,
                'level': 45,
                'description': 'CTA, w: 57, l: 45'
        },
        'CTA_w_57_l_49': {
                'window': 57,
                'level': 49,
                'description': 'CTA, w: 57, l: 49'
        },
        ##########
        # Comparison of CT and CT Angiography Source Images With Diffusion-Weighted Imaging in Patients With Acute Stroke Within 6 Hours After Onset, Schramm et al.
        'CTA_w_75_l_40': {
                'window': 75,
                'level': 40,
                'description': 'CTA, w: 75, l: 40'
        },
        ##########
        # CT Angiographic Measurement of the Carotid Artery: Optimizing Visualization by Manipulating Window and Level Settings and Contrast Material Attenuation, Kenneth et  al.
        'CTA_w_96_l_50': {
            'window': 96,
            'level': 50,
            'description': 'CTA, w: 96, l: 50 (D=250 HU)'
        },
        'CTA_w_96_l_87': {
                'window': 96,
                'level': 87,
                'description': 'CTA, w: 96, l: 87 (D=275 HU)'
        },
        'CTA_w_112_l_60': {
                'window': 112,
                'level': 60,
                'description': 'CTA, w: 112, l: 60 (D=250 HU)'
        },
        'CTA_w_112_l_100': {
                'window': 112,
                'level': 100,
                'description': 'CTA, w: 112, l: 100 (D=275 HU)'
        },
        'CTA_w_128_l_70': {
                'window': 128,
                'level': 70,
                'description': 'CTA, w: 128, l: 70 (D=250 HU)'
        },
        'CTA_w_128_l_112': {
                'window': 128,
                'level': 112,
                'description': 'CTA, w: 128, l: 112 (D=275 HU)'
        },
    }
    
    if config not in params:
        raise ValueError(f"Unknown CTA configuration: {config}")
        
    return params[config] 
