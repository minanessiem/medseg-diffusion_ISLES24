"""
Verification test for factory integration (Phase 4.3).

Run with: python tests/test_factory_integration.py
"""

import torch
import unittest
from omegaconf import OmegaConf


def _build_factory_cfg(spatial_dims: str):
    """Create 2D or 3D config for factory integration tests."""
    if spatial_dims == "2d":
        image_size = 256
        feature_size = 48
        depths = [2, 2, 2, 2]
        num_heads = [3, 6, 12, 24]
        batch_size = 2
    else:
        image_size = 64
        feature_size = 12
        depths = [1, 1, 1, 1]
        num_heads = [1, 2, 3, 4]
        batch_size = 1

    cfg = OmegaConf.create({
        'model': {
            'architecture': 'swinunetr',
            'image_size': image_size,
            'spatial_dims': spatial_dims,
            'image_channels': 2,
            'out_channels': 1,
            'feature_size': feature_size,
            'depths': depths,
            'num_heads': num_heads,
            'drop_rate': 0.0,
            'attn_drop_rate': 0.0,
        },
        'diffusion': {
            'type': 'Discriminative',
            'timesteps': 1,
        },
        'environment': {
            'device': 'cpu',
        },
        'loss': {
            'discriminative': {
                'deep_supervision': {
                    'enabled': False,
                    'final_head': 0,
                    'head_parser': 'auto',
                },
                'terms': [
                    {
                        'loss': 'DiceLoss',
                        'input_domain': 'probabilities',
                        'weight': 1.0,
                        'params': {'smooth': 1e-5, 'apply_sigmoid': False},
                        'supervision': {
                            'mode': 'final_only',
                            'heads': [0],
                            'weights': [1.0],
                        },
                    },
                    {
                        'loss': 'BCELoss',
                        'input_domain': 'probabilities',
                        'weight': 1.0,
                        'params': {'pos_weight': None, 'apply_sigmoid': False},
                        'supervision': {
                            'mode': 'final_only',
                            'heads': [0],
                            'weights': [1.0],
                        },
                    },
                ],
            }
        }
    })
    return cfg, batch_size


def _build_factory_inputs(spatial_dims: str, batch_size: int):
    """Create synthetic mask/image tensors for 2D or 3D factory tests."""
    if spatial_dims == "2d":
        mask = torch.rand(batch_size, 1, 256, 256)
        img = torch.randn(batch_size, 2, 256, 256)
    else:
        mask = torch.rand(batch_size, 1, 64, 64, 64)
        img = torch.randn(batch_size, 2, 64, 64, 64)
    return mask, img


def _run_factory_integration_case(spatial_dims: str):
    """Shared execution path for 2D/3D factory integration tests."""
    print("=" * 60)
    print(f"Factory Integration Verification Test ({spatial_dims.upper()})")
    print("=" * 60)

    # Create config
    cfg, batch_size = _build_factory_cfg(spatial_dims)

    # Test model factory
    print("\n[1/3] Testing model factory (build_model)...")
    from src.models.model_factory import build_model
    model = build_model(cfg)

    model_type = type(model).__name__
    assert model_type == 'SwinUNetRAdapter', f"Expected SwinUNetRAdapter, got {model_type}"
    print(f"  ✓ build_model returned: {model_type}")

    # Test diffusion factory
    print("\n[2/3] Testing diffusion factory (build_diffusion)...")
    from src.diffusion.diffusion import Diffusion
    adapter = Diffusion.build_diffusion(model, cfg, torch.device('cpu'))

    adapter_type = type(adapter).__name__
    assert adapter_type == 'DiscriminativeAdapter', f"Expected DiscriminativeAdapter, got {adapter_type}"
    print(f"  ✓ build_diffusion returned: {adapter_type}")

    # Test end-to-end forward pass
    print("\n[3/3] Testing end-to-end forward pass...")
    mask, img = _build_factory_inputs(spatial_dims, batch_size)
    loss, mses, t, components = adapter.forward(mask, img)
    print(f"  ✓ Forward pass successful")
    print(f"  ✓ Loss: {loss.item():.4f}")
    print(f"  ✓ Components: {list(components.keys())}")
    assert tuple(mses.shape) == (batch_size,), f"Expected sample_mses shape {(batch_size,)}, got {mses.shape}"
    assert tuple(t.shape) == (batch_size,), f"Expected timesteps shape {(batch_size,)}, got {t.shape}"
    assert "total" in components, "Missing total in loss components"
    pred = adapter.sample(img)
    assert tuple(pred.shape) == tuple(mask.shape), (
        f"Expected sample() output shape {tuple(mask.shape)}, got {tuple(pred.shape)}"
    )
    print(f"  ✓ sample() shape: {tuple(pred.shape)}")
    
    print("\n" + "=" * 60)
    print("✓ All factory integration tests passed!")
    print("=" * 60)


def test_factory_integration():
    """Test 2D factory integration path."""
    _run_factory_integration_case("2d")


def test_factory_integration_3d():
    """Test 3D factory integration path."""
    _run_factory_integration_case("3d")


if __name__ == "__main__":
    test_factory_integration()


class TestFactoryIntegration(unittest.TestCase):
    """unittest wrapper for function-based factory integration test."""

    def test_factory_integration(self):
        test_factory_integration()

    def test_factory_integration_3d(self):
        test_factory_integration_3d()

