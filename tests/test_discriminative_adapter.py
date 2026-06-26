"""
Verification test for DiscriminativeAdapter (Phase 3.2).

Run with: python tests/test_discriminative_adapter.py
"""

import torch
import torch.nn as nn
import unittest
from omegaconf import OmegaConf


def _build_adapter_cfg(spatial_dims: str):
    """Create 2D or 3D SwinUNETR config for adapter tests."""
    if spatial_dims == "2d":
        image_size = 256
        feature_size = 48
        depths = [2, 2, 2, 2]
        num_heads = [3, 6, 12, 24]
    else:
        # Lightweight 3D test footprint to keep CPU test runtime reasonable.
        image_size = 64
        feature_size = 12
        depths = [1, 1, 1, 1]
        num_heads = [1, 2, 3, 4]

    return OmegaConf.create({
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
                        'params': {'smooth': 1.0e-5, 'apply_sigmoid': False},
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


def _build_adapter_inputs(spatial_dims: str):
    """Create modality/mask tensors for 2D or 3D adapter tests."""
    if spatial_dims == "2d":
        mask = torch.rand(2, 1, 256, 256)
        img = torch.randn(2, 2, 256, 256)
    else:
        mask = torch.rand(1, 1, 64, 64, 64)
        img = torch.randn(1, 2, 64, 64, 64)
    return mask, img


def _run_discriminative_adapter_case(spatial_dims: str):
    """Shared execution path for 2D/3D DiscriminativeAdapter tests."""
    print("=" * 60)
    print(f"DiscriminativeAdapter Verification Test ({spatial_dims.upper()})")
    print("=" * 60)

    # Create full config
    cfg = _build_adapter_cfg(spatial_dims)

    # Build model and adapter
    print("\n[1/6] Building model and adapter...")
    from src.models.SwinUNetR import SwinUNetRAdapter
    from src.diffusion.discriminative_adapter import DiscriminativeAdapter

    model = SwinUNetRAdapter(cfg)
    adapter = DiscriminativeAdapter(model, cfg, device=torch.device('cpu'))

    # Verify num_timesteps attribute
    print("\n[2/6] Verifying num_timesteps attribute...")
    assert adapter.num_timesteps == 1, f"Expected num_timesteps=1, got {adapter.num_timesteps}"
    print("  ✓ num_timesteps = 1 (for logger compatibility)")

    # Test forward return type
    print("\n[3/6] Testing forward return type...")
    mask, img = _build_adapter_inputs(spatial_dims)
    expected_shape = tuple(mask.shape)

    result = adapter.forward(mask, img)
    assert len(result) == 4, f"Expected 4-tuple, got {len(result)}"
    loss, sample_mses, t, components = result

    assert loss.dim() == 0, f"Loss should be scalar, got dim={loss.dim()}"
    assert sample_mses.shape == (mask.shape[0],), f"MSEs shape: {sample_mses.shape}"
    assert t.shape == (mask.shape[0],), f"Timesteps shape: {t.shape}"
    assert isinstance(components, dict), f"Components should be dict, got {type(components)}"
    print("  ✓ Returns 4-tuple: (loss, sample_mses, t, components)")
    print(f"  ✓ Loss: {loss.item():.4f}")
    print(f"  ✓ MSEs shape: {sample_mses.shape} (dummy zeros)")
    print(f"  ✓ Timesteps shape: {t.shape} (dummy zeros)")

    # Test loss components
    print("\n[4/6] Verifying loss components...")
    assert 'DiceLoss' in components, "Missing 'DiceLoss' in components"
    assert 'BCELoss' in components, "Missing 'BCELoss' in components"
    assert 'total' in components, "Missing 'total' in components"
    print(f"  ✓ DiceLoss: {components['DiceLoss']:.4f}")
    print(f"  ✓ BCELoss: {components['BCELoss']:.4f}")
    print(f"  ✓ total: {components['total']:.4f}")

    # Test forward with intermediates
    print("\n[5/6] Testing forward with intermediates...")
    loss, mses, t, intermediates = adapter.forward(mask, img, return_intermediates=True)
    assert 'img' in intermediates, "Missing 'img' in intermediates"
    assert 'mask' in intermediates, "Missing 'mask' in intermediates"
    assert 'pred' in intermediates, "Missing 'pred' in intermediates"
    assert tuple(intermediates['pred'].shape) == expected_shape, f"Pred shape: {intermediates['pred'].shape}"
    print("  ✓ Intermediates: img, mask, pred")
    print(f"  ✓ Prediction shape: {intermediates['pred'].shape}")

    # Test sample
    print("\n[6/6] Testing sample and sample_with_snapshots...")
    pred = adapter.sample(img)
    assert tuple(pred.shape) == expected_shape, f"Sample shape: {pred.shape}"
    print(f"  ✓ sample() shape: {pred.shape}")

    # Test sample_with_snapshots
    snapshots = list(adapter.sample_with_snapshots(img))
    assert len(snapshots) == 1, f"Should yield single snapshot, got {len(snapshots)}"
    assert snapshots[0][0] == 0, f"Timestep should be 0, got {snapshots[0][0]}"
    assert tuple(snapshots[0][1].shape) == expected_shape, f"Snapshot shape: {snapshots[0][1].shape}"
    print(f"  ✓ sample_with_snapshots() yields 1 snapshot at t=0")

    print("\n" + "=" * 60)
    print("✓ All DiscriminativeAdapter tests passed!")
    print("=" * 60)


def test_discriminative_adapter():
    """Test DiscriminativeAdapter interface compatibility (2D path)."""
    _run_discriminative_adapter_case("2d")


def test_discriminative_adapter_3d():
    """Test DiscriminativeAdapter interface compatibility (3D path)."""
    _run_discriminative_adapter_case("3d")


if __name__ == "__main__":
    test_discriminative_adapter()


def test_discriminative_adapter_deep_supervision():
    """Test deep-supervision wiring path in DiscriminativeAdapter."""

    class DummyStackedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.image_channels = 1
            self.mask_channels = 1
            self.image_size = 16
            self.output_channels = 1

        def forward(self, x):
            final_head = torch.full(
                (x.shape[0], 1, x.shape[2], x.shape[3]),
                0.2,
                device=x.device,
                dtype=x.dtype,
            )
            aux_head = torch.full(
                (x.shape[0], 1, x.shape[2], x.shape[3]),
                0.8,
                device=x.device,
                dtype=x.dtype,
            )
            # Mimic DynUNet behavior:
            # - train mode: stacked deep-supervision output
            # - eval mode: final head only
            if self.training:
                return torch.stack([final_head, aux_head], dim=1)  # [B,S,C,H,W]
            return final_head

    cfg = OmegaConf.create(
        {
            "environment": {"device": "cpu"},
            "loss": {
                "discriminative": {
                    "deep_supervision": {
                        "enabled": True,
                        "final_head": 0,
                        "head_parser": "stacked",
                        "default_supervision": {
                            "mode": "weighted_heads",
                            "heads": [0, 1],
                            "weights": [1.0, 0.5],
                        },
                    },
                    "terms": [
                        {
                            "loss": "DiceLoss",
                            "input_domain": "probabilities",
                            "weight": 1.0,
                            "params": {"smooth": 1.0e-5, "apply_sigmoid": False},
                            "supervision": {"mode": "inherit_default"},
                        }
                    ],
                }
            },
        }
    )

    from src.diffusion.discriminative_adapter import DiscriminativeAdapter

    adapter = DiscriminativeAdapter(DummyStackedModel(), cfg, device=torch.device("cpu"))

    mask = torch.zeros(2, 1, 16, 16)
    img = torch.randn(2, 1, 16, 16)

    loss, sample_mses, t, components = adapter.forward(mask, img)
    assert loss.dim() == 0, f"Loss should be scalar, got dim={loss.dim()}"
    assert sample_mses.shape == (2,), f"MSEs shape: {sample_mses.shape}"
    assert t.shape == (2,), f"Timesteps shape: {t.shape}"
    assert "DiceLoss_head0" in components, "Missing DiceLoss_head0 component"
    assert "DiceLoss_head1" in components, "Missing DiceLoss_head1 component"
    assert "DiceLoss" in components, "Missing aggregated DiceLoss component"
    assert "total" in components, "Missing total component"

    _, _, _, intermediates = adapter.forward(mask, img, return_intermediates=True)
    assert intermediates["pred"].shape == (2, 1, 16, 16), (
        f"Expected final-head prediction shape (2,1,16,16), got {intermediates['pred'].shape}"
    )
    expected_probability = torch.sigmoid(torch.full_like(intermediates["pred"], 0.2))
    assert torch.allclose(intermediates["pred"], expected_probability), (
        "Expected intermediates['pred'] to map to final head probabilities."
    )

    # Inference path should work when model outputs single tensor in eval mode
    # even if deep supervision is configured.
    adapter.eval()
    pred = adapter.sample(img)
    assert pred.shape == (2, 1, 16, 16), f"Sample shape: {pred.shape}"
    assert torch.allclose(pred, torch.sigmoid(torch.full_like(pred, 0.2))), (
        "sample() should return final head probabilities."
    )

    snapshots = list(adapter.sample_with_snapshots(img))
    assert len(snapshots) == 1, f"Expected single snapshot, got {len(snapshots)}"
    assert snapshots[0][0] == 0, f"Expected timestep 0 snapshot, got {snapshots[0][0]}"
    assert torch.allclose(
        snapshots[0][1], torch.sigmoid(torch.full_like(snapshots[0][1], 0.2))
    ), "sample_with_snapshots() should return final head probabilities."


class TestDiscriminativeAdapter(unittest.TestCase):
    """unittest wrappers for function-based discriminative adapter tests."""

    def test_discriminative_adapter(self):
        test_discriminative_adapter()

    def test_discriminative_adapter_3d(self):
        test_discriminative_adapter_3d()

    def test_discriminative_adapter_deep_supervision(self):
        test_discriminative_adapter_deep_supervision()

