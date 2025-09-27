import torch
from monai.transforms import ScaleIntensityRange

def process_mtt(data: torch.Tensor, min_val: float, max_val: float,
                description: str = None, **kwargs) -> torch.Tensor:
    """Process MTT images with window/level normalization"""
    data = torch.nan_to_num(data, nan=0.0, posinf=max_val, neginf=0.0)    
    data = torch.clamp(data, 0, max_val)

    transform = ScaleIntensityRange(
        a_min=min_val,
        a_max=max_val,
        b_min=0,
        b_max=1,
        clip=True
    )
    normalized = transform(data)
    return normalized

def get_mtt_params(config: str, data_stats: dict) -> dict:
    """Get MTT-specific parameters for processing"""
    params = {
        'MTT_min_0_max_4': {
            'min_val': 0,
            'max_val': 4,
            'description': 'MTT normalized: 0 < MTT < 4'
        },
        'MTT_min_0_max_6': {
            'min_val': 0,
            'max_val': 6,
            'description': 'MTT normalized: 0 < MTT < 6'
        },
        'MTT_min_0_max_8': {
            'min_val': 0,
            'max_val': 8,
            'description': 'MTT normalized: 0 < MTT < 8'
        },
        'MTT_min_0_max_10': {
            'min_val': 0,
            'max_val': 10,
            'description': 'MTT normalized: 0 < MTT < 10'
        },
        'MTT_min_0_max_12': {
            'min_val': 0,
            'max_val': 12,
            'description': 'MTT normalized: 0 < MTT < 12'
        },
        'MTT_min_0_max_16': {
            'min_val': 0,
            'max_val': 16,
            'description': 'MTT normalized: 0 < MTT < 16'
        },
        'MTT_min_0_max_30': {
            'min_val': 0,
            'max_val': 30,
            'description': 'MTT normalized: 0 < MTT < 30'
        },
        'MTT_min_0_max_maxval': {
            'min_val': 0,
            'max_val': data_stats['max_val'],
            'description': 'MTT with direct min to max scaling (full range)'
        },
        'MTT_min_4_max_10': {
            'min_val': 4,
            'max_val': 10,
            'description': 'MTT normalized: 4 < MTT < 10'
        },
        'MTT_min_6_max_10': {
            'min_val': 6,
            'max_val': 10,
            'description': 'MTT normalized: 6 < MTT < 10'
        },
        'MTT_min_4_max_30': {
            'min_val': 4,
            'max_val': 30,
            'description': 'MTT normalized: 4 < MTT < 30'
        },
        'MTT_min_6_max_30': {
            'min_val': 6,
            'max_val': 30,
            'description': 'MTT normalized: 6 < MTT < 30'
        },
        'MTT_min_8_max_30': {
            'min_val': 8,
            'max_val': 30,
            'description': 'MTT normalized: 8 < MTT < 30'
        },
        'MTT_min_10_max_30': {
            'min_val': 10,
            'max_val': 30,
            'description': 'MTT normalized: 10 < MTT < 30'
        },
    }
    
    if config not in params:
        raise ValueError(f"Unknown MTT configuration: {config}")
        
    return params[config] 