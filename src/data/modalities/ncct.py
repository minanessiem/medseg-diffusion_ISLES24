import torch
from monai.transforms import ScaleIntensityRange

def process_ncct(data: torch.Tensor, window: float = None, level: float = None,
                use_direct_scaling: bool = False, min_val: float = None, max_val: float = None,
                description: str = None, **kwargs) -> torch.Tensor:
    """
    Process NCCT images with window/level normalization or direct min/max scaling
    
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

def get_ncct_params(config: str, data_stats: dict) -> dict:
    """Get NCCT-specific parameters for processing"""
    params = {
        'NCCT_min_minval_max_maxval': {
            'use_direct_scaling': True,
            'min_val': data_stats['min_val'],
            'max_val': data_stats['max_val'],
            'description': 'CTA with direct min to max scaling (full range)'
        },
        'NCCT_white': {
            'window': 8,
            'level': 27,
            'description': 'NCCT White matter window'
        },
        'NCCT_gray': {
            'window': 8,
            'level': 37,
            'description': 'NCCT Gray matter window'
        },
        'NCCT_wg': {
            'window': 49,
            'level': 35,
            'description': 'NCCT White and Gray matter window'
        },

        ############################
        # State-of-the-Art Imaging of Acute Stroke, Srinivasan et al.
        # "standard" window settings
        'NCCT_level_20_window_80':{
            'level': 20,
            'window': 80,
            'description': 'NCCT level 20 window 80'
        },
        # enhanced settings
        'NCCT_level_32_window_8':{
            'level': 32,
            'window': 8,
            'description': 'NCCT level 32 window 8'
        },
        ############################
        # Detection of Early Ischemic Changes in Noncontrast CT Head Improved with “Stroke Windows”, Shraddha Mainali et al.
        # "standard" window settings
        'NCCT_level_35_window_100':{
            'level': 35,
            'window': 100,
            'description': 'NCCT level 35 window 100'
        },
        # Stroke Windows
        'NCCT_level_35_window_30':{
            'level': 35,
            'window': 30,
            'description': 'NCCT level 35 window 30'
        },
        ############################
        # A quantitative symmetry-based analysis of hyperacute ischemic stroke lesions in noncontrast computed tomography, Peter et al.
        'NCCT_level_32_window_3':{
            'level': 32,
            'window': 3,
            'description': 'NCCT level 32 window 3'
        },
        'NCCT_level_32_window_5':{
            'level': 32,
            'window': 5,
            'description': 'NCCT level 32 window 5'
        },
        'NCCT_level_32_window_7':{
            'level': 32,
            'window': 7,
            'description': 'NCCT level 32 window 7'
        },
        'NCCT_level_32_window_9':{
            'level': 32,
            'window': 9,
            'description': 'NCCT level 32 window 9'
        },

        'NCCT_level_64_window_3':{
            'level': 64,
            'window': 3,
            'description': 'NCCT level 64 window 3'
        },
        'NCCT_level_64_window_5':{
            'level': 64,
            'window': 5,
            'description': 'NCCT level 64 window 5'
        },
        'NCCT_level_64_window_7':{
            'level': 64,
            'window': 7,
            'description': 'NCCT level 64 window 7'
        },
        'NCCT_level_64_window_9':{
            'level': 64,
            'window': 9,
            'description': 'NCCT level 64 window 9'
        },

        'NCCT_level_128_window_3':{
            'level': 128,
            'window': 3,
            'description': 'NCCT level 128 window 3'
        },
        'NCCT_level_128_window_5':{
            'level': 128,
            'window': 5,
            'description': 'NCCT level 128 window 5'
        },
        'NCCT_level_128_window_7':{
            'level': 128,
            'window': 7,
            'description': 'NCCT level 128 window 7'
        },
        'NCCT_level_128_window_9':{
            'level': 128,
            'window': 9,
            'description': 'NCCT level 128 window 9'
        },
        ############################
        # The Effect of Window Width and Window-level Settings in Non-enhanced Head CT to Increase the Diagnostic Value of Subacute Ischemic Stroke, Muqmiroh et al.
        'NCCT_level_25_window_35':{
            'level': 25,
            'window': 35,
            'description': 'NCCT level 25 window 35'
        },
        ############################
        # Can the Ischemic Penumbra Be Identified on Noncontrast CT of Acute Stroke?, Muir et al.
        'NCCT_level_40_window_80':{
            'level': 40,
            'window': 80,
            'description': 'NCCT level 40 window 80'
        },
        ############################
        # Unenhanced Computed Tomography, Camargo et al.
        # standard settings
        # 'NCCT_level_20_window_80':{
        #     'level': 20,
        #     'window': 80,
        #     'description': 'NCCT level 20 window 80'
        # },
        # narrow, nonstandard review settings
        # 'NCCT_level_32_window_8':{
        #     'level': 32,
        #     'window': 8,
        #     'description': 'NCCT level 32 window 8'
        # },
        ############################
        # Computed Tomography Angiography and Computed Tomography Perfusion in Ischemic Stroke: A Step-by-Step Approach to Image Acquisition and Three-Dimensional Postprocessing, Pomerantz et al.
        'NCCT_level_35_window_5':{
            'level': 35,
            'window': 5,
            'description': 'NCCT level 35 window 5'
        },
        'NCCT_level_35_window_10':{
            'level': 35,
            'window': 10,
            'description': 'NCCT level 35 window 10'
        },
        'NCCT_level_35_window_15':{
            'level': 35,
            'window': 15,
            'description': 'NCCT level 35 window 15'
        },
        'NCCT_level_35_window_20':{
            'level': 35,
            'window': 20,
            'description': 'NCCT level 35 window 20'
        },
        'NCCT_level_35_window_25':{
            'level': 35,
            'window': 25,
            'description': 'NCCT level 35 window 25'
        },
        'NCCT_level_35_window_30':{
            'level': 35,
            'window': 30,
            'description': 'NCCT level 35 window 30'
        },
        'NCCT_level_35_window_35':{
            'level': 35,
            'window': 35,
            'description': 'NCCT level 35 window 35'
        },
        ############################
        # CT Imaging of Acute Ischemic Stroke, Byrnes et al.
        'NCCT_level_35_window_40':{
            'level': 35,
            'window': 40,
            'description': 'NCCT level 35 window 40'
        },
        'NCCT_level_40_window_8':{
            'level': 40,
            'window': 8,
            'description': 'NCCT level 40 window 8'
        },
        'NCCT_level_40_window_16':{
            'level': 40,
            'window': 16,
            'description': 'NCCT level 40 window 16'
        },
        'NCCT_level_40_window_24':{
            'level': 40,
            'window': 24,
            'description': 'NCCT level 40 window 24'
        },
        'NCCT_level_40_window_32':{
            'level': 40,
            'window': 32,
            'description': 'NCCT level 40 window 32'
        },
        'NCCT_level_40_window_40':{
            'level': 40,
            'window': 40,
            'description': 'NCCT level 40 window 40'
        },
        #############################
        # self exploration
        'NCCT_level_20_window_70':{
            'level': 20,
            'window': 70,
            'description': 'NCCT level 20 window 70'
        },
        'NCCT_level_20_window_60':{
            'level': 20,
            'window': 60,
            'description': 'NCCT level 20 window 60'
        },
        'NCCT_level_20_window_50':{
            'level': 20,
            'window': 50,
            'description': 'NCCT level 20 window 50'
        },
        'NCCT_level_20_window_40':{
            'level': 20,
            'window': 40,
            'description': 'NCCT level 20 window 40'
        },
        'NCCT_level_20_window_30':{
            'level': 20,
            'window': 30,
            'description': 'NCCT level 20 window 30'
        },
        'NCCT_level_40_window_70':{
            'level': 40,
            'window': 70,
            'description': 'NCCT level 40 window 70'
        },
        'NCCT_level_40_window_60':{
            'level': 40,
            'window': 60,
            'description': 'NCCT level 40 window 60'
        },
        'NCCT_level_40_window_50':{
            'level': 40,
            'window': 50,
            'description': 'NCCT level 40 window 50'
        },
        'NCCT_level_40_window_40':{
            'level': 40,
            'window': 40,
            'description': 'NCCT level 40 window 40'
        },
        'NCCT_level_40_window_30':{
            'level': 40,
            'window': 30,
            'description': 'NCCT level 40 window 30'
        },
    }
    
    if config not in params:
        raise ValueError(f"Unknown NCCT configuration: {config}")
        
    return params[config] 