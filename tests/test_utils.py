"""
Test script for DiffSwinTr utilities.
Run with: python -m tests.test_utils
"""
import torch
from src.models.DiffSwinTr.utils import (
    fuse_cem_features,
    convert_nhwc_to_nchw,
    convert_nchw_to_nhwc,
)

print("=" * 60)
print("Testing DiffSwinTr Utilities")
print("=" * 60)

# Test 1: Format conversion NHWC → NCHW
print("\n[Test 1] Convert NHWC to NCHW")
x_nhwc = torch.randn(2, 64, 64, 96)
x_nchw = convert_nhwc_to_nchw(x_nhwc)
print(f"  Input (NHWC): {x_nhwc.shape}")
print(f"  Output (NCHW): {x_nchw.shape}")
assert x_nchw.shape == (2, 96, 64, 64), f"Shape mismatch: {x_nchw.shape}"
print("  ✓ PASSED")

# Test 2: Format conversion NCHW → NHWC
print("\n[Test 2] Convert NCHW to NHWC")
x_nchw = torch.randn(2, 96, 64, 64)
x_nhwc = convert_nchw_to_nhwc(x_nchw)
print(f"  Input (NCHW): {x_nchw.shape}")
print(f"  Output (NHWC): {x_nhwc.shape}")
assert x_nhwc.shape == (2, 64, 64, 96), f"Shape mismatch: {x_nhwc.shape}"
print("  ✓ PASSED")

# Test 3: Round-trip conversion
print("\n[Test 3] Round-trip conversion (NHWC → NCHW → NHWC)")
x_original = torch.randn(2, 32, 32, 192)
x_converted = convert_nchw_to_nhwc(convert_nhwc_to_nchw(x_original))
print(f"  Original: {x_original.shape}")
print(f"  After round-trip: {x_converted.shape}")
assert x_converted.shape == x_original.shape, "Shape changed!"
assert torch.allclose(x_converted, x_original), "Values changed!"
print("  ✓ PASSED")

# Test 4: Fuse CEM features with addition
print("\n[Test 4] Fuse CEM features (addition mode)")
enc_feat = torch.randn(2, 32, 32, 192)  # NHWC
cem_feat = torch.randn(2, 192, 32, 32)  # NCHW
fused = fuse_cem_features(enc_feat, cem_feat, "add")
print(f"  Encoder features (NHWC): {enc_feat.shape}")
print(f"  CEM features (NCHW): {cem_feat.shape}")
print(f"  Fused features: {fused.shape}")
assert fused.shape == (2, 32, 32, 192), f"Shape mismatch: {fused.shape}"
assert fused.shape == enc_feat.shape, "Output format should match encoder format"
print("  ✓ PASSED")

# Test 5: Verify fusion values (addition)
print("\n[Test 5] Verify fusion arithmetic (addition)")
enc_feat = torch.ones(2, 16, 16, 96) * 1.0
cem_feat = torch.ones(2, 96, 16, 16) * 2.0
fused = fuse_cem_features(enc_feat, cem_feat, "add")
expected_value = 3.0  # 1.0 + 2.0
actual_value = fused[0, 0, 0, 0].item()
print(f"  Encoder value: 1.0")
print(f"  CEM value: 2.0")
print(f"  Fused value: {actual_value}")
assert abs(actual_value - expected_value) < 1e-5, "Addition not working correctly!"
print("  ✓ PASSED")

# Test 6: Concatenation mode raises NotImplementedError
print("\n[Test 6] Concatenation mode raises NotImplementedError")
enc_feat = torch.randn(2, 32, 32, 192)
cem_feat = torch.randn(2, 192, 32, 32)
try:
    fused = fuse_cem_features(enc_feat, cem_feat, "concat")
    assert False, "Should have raised NotImplementedError!"
except NotImplementedError as e:
    print(f"  ✓ Correctly raised NotImplementedError")
    print(f"  Message: {str(e)[:60]}...")
print("  ✓ PASSED")

# Test 7: Unknown fusion mode raises ValueError
print("\n[Test 7] Unknown fusion mode raises ValueError")
enc_feat = torch.randn(2, 32, 32, 192)
cem_feat = torch.randn(2, 192, 32, 32)
try:
    fused = fuse_cem_features(enc_feat, cem_feat, "invalid_mode")
    assert False, "Should have raised ValueError!"
except ValueError as e:
    print(f"  ✓ Correctly raised ValueError")
    print(f"  Message: {str(e)[:60]}...")
print("  ✓ PASSED")

# Test 8: Fusion gradient flow
print("\n[Test 8] Fusion gradient flow")
enc_feat = torch.randn(2, 32, 32, 192, requires_grad=True)
cem_feat = torch.randn(2, 192, 32, 32)
fused = fuse_cem_features(enc_feat, cem_feat, "add")
loss = fused.sum()
loss.backward()
assert enc_feat.grad is not None, "Gradient not computed!"
assert enc_feat.grad.abs().sum() > 0, "Zero gradients!"
print(f"  Gradient sum: {enc_feat.grad.abs().sum().item():.4f}")
print("  ✓ PASSED")

# Test 9: Multiple scale fusion
print("\n[Test 9] Fuse features at multiple scales")
scales = [
    (64, 64, 96),    # Stage 1
    (32, 32, 192),   # Stage 2
    (16, 16, 384),   # Stage 3
    (8, 8, 768),     # Stage 4
]

for h, w, c in scales:
    enc_feat = torch.randn(2, h, w, c)  # NHWC
    cem_feat = torch.randn(2, c, h, w)  # NCHW
    fused = fuse_cem_features(enc_feat, cem_feat, "add")
    assert fused.shape == (2, h, w, c), f"Shape mismatch at scale {h}×{w}"
    print(f"  Scale [{h}×{w}, {c}ch] → {fused.shape} ✓")

print("  ✓ PASSED")

# Test 10: Format conversion preserves values
print("\n[Test 10] Format conversion preserves values")
x_nhwc = torch.randn(2, 32, 32, 96)

# Convert to NCHW and back
x_nchw = convert_nhwc_to_nchw(x_nhwc)
x_back = convert_nchw_to_nhwc(x_nchw)

# Check shapes
assert x_back.shape == x_nhwc.shape, "Shape not preserved!"

# Check values
assert torch.allclose(x_back, x_nhwc), "Values not preserved!"

# Check specific values
original_val = x_nhwc[0, 0, 0, 0].item()
converted_val = x_back[0, 0, 0, 0].item()
print(f"  Original [0,0,0,0]: {original_val:.6f}")
print(f"  After conversion:   {converted_val:.6f}")
print(f"  Difference: {abs(original_val - converted_val):.10f}")

print("  ✓ PASSED")

print("\n" + "=" * 60)
print("ALL UTILITY TESTS PASSED! ✓")
print("=" * 60)

