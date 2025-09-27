from typing import Tuple, Dict
from .ncct import get_ncct_params
from .cta import get_cta_params
from .cbf import get_cbf_params
from .cbv import get_cbv_params
from .mtt import get_mtt_params
from .tmax import get_tmax_params

def get_modality_params(modality_config: str, data_stats: dict) -> Tuple[str, Dict]:
    """
    Get modality-specific parameters for processing based on configuration.
    
    Args:
        modality_config: The modality configuration (e.g., 'NCCT_white', 'NCCT_gray')
        data_stats: Dictionary containing basic data statistics
    
    Returns:
        Tuple of (base_modality, parameters)
    """
    base_modality = modality_config.split('_')[0]
    
    # Route to appropriate parameter getter based on base modality
    if base_modality == 'NCCT':
        params = get_ncct_params(modality_config, data_stats)
    elif base_modality == 'CTA':
        params = get_cta_params(modality_config, data_stats)
    elif base_modality == 'CBF':
        params = get_cbf_params(modality_config, data_stats)
    elif base_modality == 'CBV':
        params = get_cbv_params(modality_config, data_stats)
    elif base_modality == 'MTT':
        params = get_mtt_params(modality_config, data_stats)
    elif base_modality == 'TMAX':
        params = get_tmax_params(modality_config, data_stats)
    else:
        raise ValueError(f"Unknown base modality: {base_modality}")
    
    return base_modality, params 