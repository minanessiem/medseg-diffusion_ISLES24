"""
Test script for time conditioning components.
Run with: python test_time_conditioning.py
"""
import torch
from src.models.DiffSwinTr.time_conditioning import TimestepEmbedder, modulate

print("=" * 60)
print("Testing Time Conditioning Components")
print("=" * 60)

# Test 1: TimestepEmbedder output shape
print("\n[Test 1] TimestepEmbedder output shape")
embedder = TimestepEmbedder(hidden_size=256)
t = torch.randint(0, 1000, (4,))
emb = embedder(t)
print(f"  Input shape: {t.shape}")
print(f"  Output shape: {emb.shape}")
assert emb.shape == (4, 256), f"Expected (4, 256), got {emb.shape}"
print("  ✓ PASSED")

# Test 2: Different timesteps produce different embeddings
print("\n[Test 2] Different timesteps produce different embeddings")
t1 = torch.tensor([0])
t2 = torch.tensor([999])
emb1 = embedder(t1)
emb2 = embedder(t2)
assert not torch.allclose(emb1, emb2), "Same embeddings for different timesteps!"
print("  ✓ PASSED - Embeddings are different")

# Test 3: Modulate with 4D tensors (Swin format)
print("\n[Test 3] Modulate with 4D tensors [B, H, W, C]")
x = torch.randn(2, 16, 16, 96)
shift = torch.randn(2, 96)
scale = torch.randn(2, 96)
out = modulate(x, shift, scale)
print(f"  Input shape: {x.shape}")
print(f"  Output shape: {out.shape}")
assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
print("  ✓ PASSED")

# Test 4: Modulate with 3D tensors (standard transformer format)
print("\n[Test 4] Modulate with 3D tensors [B, N, C]")
x = torch.randn(2, 64, 96)
shift = torch.randn(2, 96)
scale = torch.randn(2, 96)
out = modulate(x, shift, scale)
print(f"  Input shape: {x.shape}")
print(f"  Output shape: {out.shape}")
assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
print("  ✓ PASSED")

# Test 5: Gradient flow
print("\n[Test 5] Gradient flow through TimestepEmbedder")
t = torch.randint(0, 1000, (4,)).float()
t.requires_grad = False  # Timesteps don't need gradients
embedder_test = TimestepEmbedder(hidden_size=256)
emb = embedder_test(t)
loss = emb.sum()
loss.backward()
# Check that MLP weights have gradients
has_grad = embedder_test.mlp[0].weight.grad is not None
print(f"  MLP weights have gradients: {has_grad}")
assert has_grad, "Gradients not computed!"
print("  ✓ PASSED")

# Test 6: Modulate gradient flow
print("\n[Test 6] Gradient flow through modulate")
x = torch.randn(2, 16, 16, 96, requires_grad=True)
shift = torch.randn(2, 96)
scale = torch.randn(2, 96)
out = modulate(x, shift, scale)
loss = out.sum()
loss.backward()
assert x.grad is not None, "Gradient not computed for input!"
print("  ✓ PASSED")

print("\n" + "=" * 60)
print("ALL TESTS PASSED! ✓")
print("=" * 60)

