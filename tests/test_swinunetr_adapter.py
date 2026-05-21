"""
Verification test for SwinUNetR adapter (Phase 2.3).

Run with: python tests/test_swinunetr_adapter.py
"""

import torch
from omegaconf import OmegaConf


def _build_cfg(spatial_dims: str, image_size: int = 32):
    """Create minimal config for one adapter test case."""
    return OmegaConf.create(
        {
            "model": {
                "architecture": "swinunetr",
                "spatial_dims": spatial_dims,
                "image_size": image_size,
                "image_channels": 2,  # e.g., ISLES24 CBF+TMAX
                "out_channels": 1,
                "feature_size": 24,
                "depths": [2, 2, 2, 2],
                "num_heads": [1, 2, 3, 4],
                "drop_rate": 0.0,
                "attn_drop_rate": 0.0,
            }
        }
    )


def _run_case(spatial_dims: str):
    """Run one SwinUNetR adapter shape/range case for 2D or 3D."""
    from src.models.SwinUNetR import SwinUNetRAdapter

    expected_spatial_dims = 2 if spatial_dims == "2d" else 3
    expected_img_size = (32, 32) if expected_spatial_dims == 2 else (32, 32, 32)
    forward_image_size = 64
    forward_expected_img_size = (
        (forward_image_size, forward_image_size)
        if expected_spatial_dims == 2
        else (forward_image_size, forward_image_size, forward_image_size)
    )

    # Shape-expansion validation at the target debug size (32).
    cfg_shape = _build_cfg(spatial_dims, image_size=32)
    model_shape = SwinUNetRAdapter(cfg_shape)

    assert model_shape.spatial_dims == expected_spatial_dims, (
        f"Expected spatial_dims={expected_spatial_dims}, got {model_shape.spatial_dims}"
    )
    assert model_shape.image_size == 32, f"Expected image_size=32, got {model_shape.image_size}"
    assert model_shape.expanded_img_size == expected_img_size, (
        f"Expected expanded_img_size={expected_img_size}, got {model_shape.expanded_img_size}"
    )
    assert model_shape.image_channels == 2, (
        f"Expected image_channels=2, got {model_shape.image_channels}"
    )
    assert model_shape.mask_channels == 1, (
        f"Expected mask_channels=1, got {model_shape.mask_channels}"
    )
    assert model_shape.output_channels == 1, (
        f"Expected output_channels=1, got {model_shape.output_channels}"
    )
    print(f"  ✓ spatial_dims = {model_shape.spatial_dims}")
    print(f"  ✓ image_size = {model_shape.image_size}")
    print(f"  ✓ expanded_img_size = {model_shape.expanded_img_size}")
    print("  ✓ image_channels = 2")
    print("  ✓ mask_channels = 1")
    print("  ✓ output_channels = 1")

    # Forward validation at a stable size to avoid 1x1/1x1x1 InstanceNorm edge cases.
    cfg_forward = _build_cfg(spatial_dims, image_size=forward_image_size)
    model_forward = SwinUNetRAdapter(cfg_forward)
    model_forward.eval()

    print(f"  ✓ Testing forward pass at image_size={forward_image_size}...")
    assert model_forward.expanded_img_size == forward_expected_img_size, (
        f"Expected expanded_img_size={forward_expected_img_size}, "
        f"got {model_forward.expanded_img_size}"
    )

    if expected_spatial_dims == 2:
        x = torch.randn(1, 2, forward_image_size, forward_image_size)
        expected_shape = (1, 1, forward_image_size, forward_image_size)
    else:
        x = torch.randn(1, 2, forward_image_size, forward_image_size, forward_image_size)
        expected_shape = (1, 1, forward_image_size, forward_image_size, forward_image_size)

    with torch.no_grad():
        out = model_forward(x)

    assert out.shape == expected_shape, f"Expected shape {expected_shape}, got {out.shape}"
    print(f"  ✓ Input shape: {tuple(x.shape)}")
    print(f"  ✓ Output shape: {tuple(out.shape)}")

    print("  ✓ Verifying output range [0, 1]...")
    assert out.min() >= 0, f"Output min {out.min():.4f} < 0 (sigmoid not applied?)"
    assert out.max() <= 1, f"Output max {out.max():.4f} > 1 (sigmoid not applied?)"
    print(f"  ✓ Output range: [{out.min():.4f}, {out.max():.4f}]")


def test_swinunetr_adapter():
    """Test SwinUNetR adapter properties, forward pass, and output range for 2D and 3D."""
    print("=" * 60)
    print("SwinUNetR Adapter Verification Test (2D + 3D)")
    print("=" * 60)

    print("\n[1/2] Testing 2D case...")
    _run_case("2d")

    print("\n[2/2] Testing 3D case...")
    _run_case("3d")

    print("\n" + "=" * 60)
    print("✓ All SwinUNetR adapter tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_swinunetr_adapter()

