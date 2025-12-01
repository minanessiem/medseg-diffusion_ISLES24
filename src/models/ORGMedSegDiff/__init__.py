"""
Official MedSegDiff architecture integration.

This package contains the official MedSegDiff model code with minimal
modifications (import path changes only). The original architecture
is wrapped by ORGMedSegDiffAdapter for pipeline compatibility.

Components:
    - unet.py: Official UNet models (UNetModel_newpreview, UNetModel_v1preview)
    - utils.py: Helper functions (sigmoid_helper, InitWeights_He, etc.)
    - fp16_util.py: FP16 conversion utilities
    - model_adapter.py: Thin wrapper for pipeline integration

Architecture Variants:
    - UNetModel_newpreview: "new" version with anchor-based highway network
    - UNetModel_v1preview: "v1" version with encoder-feature highway network

Usage:
    # Via model factory (recommended)
    model = build_model(cfg)  # with cfg.model.architecture = "org_medsegdiff"
    
    # Direct instantiation
    from src.models.ORGMedSegDiff import ORGMedSegDiffAdapter
    model = ORGMedSegDiffAdapter(cfg)

Key Features:
    - Highway network produces calibration output (auxiliary segmentation)
    - Calibration output has sigmoid applied internally (via sigmoid_helper)
    - Returns tuple: (noise_prediction, calibration_output)

Notes:
    - Grafted from official MedSegDiff repository with minimal modifications
    - Import paths changed to use src.improved_diffusion.nn
    - batchgenerators dependency stubbed (not needed for training/sampling)
"""

# Main adapter for pipeline integration
from .model_adapter import ORGMedSegDiffAdapter

# Raw model classes for inspection/debugging
from .unet import UNetModel_newpreview, UNetModel_v1preview

__all__ = [
    'ORGMedSegDiffAdapter',
    'UNetModel_newpreview',
    'UNetModel_v1preview',
]

