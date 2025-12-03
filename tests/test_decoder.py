"""
Test script for decoder components.
Run with: python -m tests.test_decoder
"""
import torch
from src.models.DiffSwinTr.decoder import PatchExpand, FinalPatchExpand

print("=" * 60)
print("Testing Decoder Components")
print("=" * 60)

# Test 1: PatchExpand shape transformation
print("\n[Test 1] PatchExpand shape transformation")
expand = PatchExpand(dim=192, dim_scale=2)
x = torch.randn(2, 16, 16, 192)
out = expand(x)
print(f"  Input shape: {x.shape}")
print(f"  Output shape: {out.shape}")
print(f"  Expected: [2, 32, 32, 96]")
assert out.shape == (2, 32, 32, 96), f"Expected (2, 32, 32, 96), got {out.shape}"
print("  ✓ PASSED")

# Test 2: PatchExpand gradient flow
print("\n[Test 2] PatchExpand gradient flow")
expand = PatchExpand(dim=192, dim_scale=2)
x = torch.randn(2, 16, 16, 192, requires_grad=True)
out = expand(x)
loss = out.sum()
loss.backward()
assert x.grad is not None, "Gradient not computed!"
assert x.grad.abs().sum() > 0, "Zero gradients!"
print(f"  Gradient sum: {x.grad.abs().sum().item():.4f}")
print("  ✓ PASSED")

# Test 3: PatchExpand with different dimensions
print("\n[Test 3] PatchExpand with different input dimensions")
test_cases = [
    (384, 2, 16, 16, (2, 32, 32, 192)),  # 384 → 192
    (768, 2, 8, 8, (2, 16, 16, 384)),    # 768 → 384
    (96, 2, 64, 64, (2, 128, 128, 48)),  # 96 → 48
]
for dim, scale, h, w, expected_shape in test_cases:
    expand = PatchExpand(dim=dim, dim_scale=scale)
    x = torch.randn(2, h, w, dim)
    out = expand(x)
    print(f"  [{dim}, {h}, {w}] → {out.shape}")
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
print("  ✓ PASSED")

# Test 4: FinalPatchExpand shape transformation
print("\n[Test 4] FinalPatchExpand shape transformation")
final_expand = FinalPatchExpand(dim=96, patch_size=4)
x = torch.randn(2, 64, 64, 96)
out = final_expand(x)
print(f"  Input shape: {x.shape}")
print(f"  Output shape: {out.shape}")
print(f"  Expected: [2, 256, 256, 96]")
assert out.shape == (2, 256, 256, 96), f"Expected (2, 256, 256, 96), got {out.shape}"
print("  ✓ PASSED")

# Test 5: FinalPatchExpand gradient flow
print("\n[Test 5] FinalPatchExpand gradient flow")
final_expand = FinalPatchExpand(dim=96, patch_size=4)
x = torch.randn(2, 64, 64, 96, requires_grad=True)
out = final_expand(x)
loss = out.sum()
loss.backward()
assert x.grad is not None, "Gradient not computed!"
assert x.grad.abs().sum() > 0, "Zero gradients!"
print(f"  Gradient sum: {x.grad.abs().sum().item():.4f}")
print("  ✓ PASSED")

# Test 6: FinalPatchExpand with different patch sizes
print("\n[Test 6] FinalPatchExpand with different patch sizes")
test_cases = [
    (96, 2, 128, 128, (2, 256, 256, 96)),   # 2x expansion
    (96, 4, 64, 64, (2, 256, 256, 96)),     # 4x expansion
    (128, 4, 64, 64, (2, 256, 256, 128)),   # 4x expansion, different dim
]
for dim, patch_size, h, w, expected_shape in test_cases:
    final_expand = FinalPatchExpand(dim=dim, patch_size=patch_size)
    x = torch.randn(2, h, w, dim)
    out = final_expand(x)
    print(f"  [{dim}, {h}, {w}] (patch={patch_size}) → {out.shape}")
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
print("  ✓ PASSED")

# Test 7: Verify channel dimension is preserved in FinalPatchExpand
print("\n[Test 7] FinalPatchExpand preserves channel dimension")
for dim in [64, 96, 128]:
    final_expand = FinalPatchExpand(dim=dim, patch_size=4)
    x = torch.randn(2, 64, 64, dim)
    out = final_expand(x)
    assert out.shape[-1] == dim, f"Channel dim changed: {dim} → {out.shape[-1]}"
    print(f"  Input channels: {dim}, Output channels: {out.shape[-1]} ✓")
print("  ✓ PASSED")

# Test 8: Verify channel halving in PatchExpand
print("\n[Test 8] PatchExpand halves channel dimension (dim_scale=2)")
for dim in [96, 192, 384, 768]:
    expand = PatchExpand(dim=dim, dim_scale=2)
    x = torch.randn(2, 16, 16, dim)
    out = expand(x)
    expected_channels = dim // 2
    assert out.shape[-1] == expected_channels, \
        f"Channel dim mismatch: expected {expected_channels}, got {out.shape[-1]}"
    print(f"  Input channels: {dim}, Output channels: {out.shape[-1]} ✓")
print("  ✓ PASSED")

print("\n" + "=" * 60)
print("ALL DECODER TESTS PASSED! ✓")
print("=" * 60)

