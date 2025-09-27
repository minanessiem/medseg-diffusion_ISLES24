import torch
from monai.transforms import ScaleIntensityRange

def process_cbv(data: torch.Tensor, min_val: float, max_val: float,
                description: str = None, **kwargs) -> torch.Tensor:
    """Process CBV images with min/max normalization"""
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



def get_cbv_params(config: str, data_stats: dict) -> dict:
    """Get CBV-specific parameters for processing"""
    params = {
        # Normalization configurations
        'CBV_min_0_max_1.8': {
            'min_val': 0,
            'max_val': 1.8,
            'ROI_mask': False,
            'description': 'CBV normalized: 0 < CBV < 1.8 mL/100g'
        },
        'CBV_min_0_max_1.9': {
            'min_val': 0,
            'max_val': 1.9,
            'ROI_mask': False,
            'description': 'CBV normalized: 0 < CBV < 1.9 mL/100g'
        },
        'CBV_min_0_max_2': {
            'min_val': 0,
            'max_val': 2,
            'ROI_mask': False,
            'description': 'CBV normalized: 0 < CBV < 2.0 mL/100g'
        },
        'CBV_min_0_max_2.2': {
            'min_val': 0,
            'max_val': 2.2,
            'ROI_mask': False,
            'description': 'CBV normalized: 0 < CBV < 2.2 mL/100g'
        },
        'CBV_min_0_max_4': {
            'min_val': 0,
            'max_val': 4,
            'ROI_mask': False,
            'description': 'CBV normalized: 0 < CBV < 4.0 mL/100g'
        },
        'CBV_min_0_max_6': {
            'min_val': 0,
            'max_val': 6,
            'ROI_mask': False,
            'description': 'CBV normalized: 0 < CBV < 6.0 mL/100g'
        },
        'CBV_min_0_max_9': {
            'min_val': 0,
            'max_val': 9,
            'ROI_mask': False,
            'description': 'CBV normalized: 0 < CBV < 9.0 mL/100g'
        },
        'CBV_min_0_max_maxval': {
            'min_val': 0,
            'max_val': data_stats['max_val'],
            'ROI_mask': False,
            'description': 'CBV normalized: 0 < CBV < max value'
        }
    }
    
    if config not in params:
        raise ValueError(f"Unknown CBV configuration: {config}")
        
    return params[config] 
