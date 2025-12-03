"""
Test script for Conditional Encoder Module (CEM).
Run with: python -m tests.test_conditional_encoder
"""
import torch
from src.models.DiffSwinTr.conditional_encoder import (
    ConditionalEncoderModule, 
    FeatureExtractionModule
)

print("=" * 60)
print("Testing Conditional Encoder Module (CEM)")
print("=" * 60)

# Test 1: FeatureExtractionModule output shape
print("\n[Test 1] FeatureExtractionModule shape")
fem = FeatureExtractionModule(96, 192)
x = torch.randn(2, 96, 32, 32)
out = fem(x)
print(f"  Input shape: {x.shape}")
print(f"  Output shape: {out.shape}")
print(f"  Expected: [2, 192, 32, 32]")
assert out.shape == (2, 192, 32, 32), f"Shape mismatch: {out.shape}"
print("  ✓ PASSED")

# Test 2: FeatureExtractionModule gradient flow
print("\n[Test 2] FeatureExtractionModule gradient flow")
fem = FeatureExtractionModule(96, 192)
x = torch.randn(2, 96, 32, 32, requires_grad=True)
out = fem(x)
loss = out.sum()
loss.backward()
assert x.grad is not None, "Gradient not computed!"
assert x.grad.abs().sum() > 0, "Zero gradients!"
print(f"  Gradient sum: {x.grad.abs().sum().item():.4f}")
print("  ✓ PASSED")

# Test 3: ConditionalEncoderModule multi-scale outputs
print("\n[Test 3] CEM multi-scale output shapes")
cem = ConditionalEncoderModule(in_channels=2, embed_dim=96)
x = torch.randn(2, 2, 256, 256)
features = cem(x)

print(f"  Input shape: {x.shape}")
print(f"  Number of feature scales: {len(features)}")
assert len(features) == 4, f"Expected 4 features, got {len(features)}"

expected_shapes = [
    (2, 96, 64, 64),    # Stage 1: embed_dim, H/4, W/4
    (2, 192, 32, 32),   # Stage 2: 2*embed_dim, H/8, W/8
    (2, 384, 16, 16),   # Stage 3: 4*embed_dim, H/16, W/16
    (2, 768, 8, 8),     # Stage 4: 8*embed_dim, H/32, W/32
]

for i, (feat, expected) in enumerate(zip(features, expected_shapes)):
    print(f"  f{i+1}: {feat.shape} (expected {expected})")
    assert feat.shape == expected, f"f{i+1} shape mismatch!"

print("  ✓ PASSED")

# Test 4: CEM with different embed dimensions
print("\n[Test 4] CEM with different embed_dim values")
test_cases = [
    (64, [(2, 64, 64, 64), (2, 128, 32, 32), (2, 256, 16, 16), (2, 512, 8, 8)]),
    (96, [(2, 96, 64, 64), (2, 192, 32, 32), (2, 384, 16, 16), (2, 768, 8, 8)]),
    (128, [(2, 128, 64, 64), (2, 256, 32, 32), (2, 512, 16, 16), (2, 1024, 8, 8)]),
]

for embed_dim, expected_shapes in test_cases:
    cem = ConditionalEncoderModule(in_channels=2, embed_dim=embed_dim)
    x = torch.randn(2, 2, 256, 256)
    features = cem(x)
    
    print(f"  embed_dim={embed_dim}:")
    for i, (feat, expected) in enumerate(zip(features, expected_shapes)):
        assert feat.shape == expected, f"Mismatch at stage {i+1}: {feat.shape} vs {expected}"
        print(f"    Stage {i+1}: {feat.shape} ✓")

print("  ✓ PASSED")

# Test 5: CEM with different input channels
print("\n[Test 5] CEM with different input channel counts")
for in_chans in [1, 2, 3, 4]:
    cem = ConditionalEncoderModule(in_channels=in_chans, embed_dim=96)
    x = torch.randn(2, in_chans, 256, 256)
    features = cem(x)
    
    # Check that all features are produced correctly
    assert len(features) == 4, f"Expected 4 features for {in_chans} input channels"
    assert features[0].shape == (2, 96, 64, 64), f"f1 shape incorrect"
    print(f"  Input channels: {in_chans} → 4 features ✓")

print("  ✓ PASSED")

# Test 6: CEM gradient flow
print("\n[Test 6] CEM gradient flow")
cem = ConditionalEncoderModule(in_channels=2, embed_dim=96)
x = torch.randn(2, 2, 256, 256, requires_grad=True)
features = cem(x)

# Compute loss from all feature scales
loss = sum(f.sum() for f in features)
loss.backward()

assert x.grad is not None, "Gradient not computed!"
assert x.grad.abs().sum() > 0, "Zero gradients!"
print(f"  Gradient sum: {x.grad.abs().sum().item():.4f}")
print("  ✓ PASSED")

# Test 7: Verify channel progression (C → 2C → 4C → 8C)
print("\n[Test 7] CEM channel progression verification")
embed_dim = 96
cem = ConditionalEncoderModule(in_channels=2, embed_dim=embed_dim)
x = torch.randn(2, 2, 256, 256)
features = cem(x)

expected_channels = [embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8]
actual_channels = [f.shape[1] for f in features]

print(f"  Base embed_dim: {embed_dim}")
print(f"  Expected channels: {expected_channels}")
print(f"  Actual channels:   {actual_channels}")

assert actual_channels == expected_channels, "Channel progression mismatch!"
print("  ✓ PASSED")

# Test 8: Verify spatial progression (H/4 → H/8 → H/16 → H/32)
print("\n[Test 8] CEM spatial progression verification")
for img_size in [128, 256, 512]:
    cem = ConditionalEncoderModule(in_channels=2, embed_dim=96)
    x = torch.randn(2, 2, img_size, img_size)
    features = cem(x)
    
    expected_spatial = [
        img_size // 4,   # H/4
        img_size // 8,   # H/8
        img_size // 16,  # H/16
        img_size // 32,  # H/32
    ]
    actual_spatial = [f.shape[2] for f in features]  # Heights
    
    print(f"  Image size {img_size}:")
    print(f"    Expected spatial: {expected_spatial}")
    print(f"    Actual spatial:   {actual_spatial}")
    assert actual_spatial == expected_spatial, f"Spatial mismatch for size {img_size}"

print("  ✓ PASSED")

print("\n" + "=" * 60)
print("ALL CEM TESTS PASSED! ✓")
print("=" * 60)

