"""
Test script for SwinUNet core architecture.
Run with: python -m tests.test_swin_unet
"""
import torch
from src.models.DiffSwinTr.swin_unet import SwinUNet

print("=" * 60)
print("Testing SwinUNet Architecture")
print("=" * 60)

# Test 1: Basic forward pass without CEM
print("\n[Test 1] Basic forward pass (no CEM)")
model = SwinUNet(
    img_size=256,
    in_chans=3,
    out_chans=1,
    embed_dim=96,
    depths=[2, 2, 2, 2],  # Smaller for faster testing
    num_heads=[3, 6, 12, 24],
    window_size=8,
    cem_enabled=False,
)

x = torch.randn(2, 3, 256, 256)
time_emb = torch.randn(2, 256)

print(f"  Input shape: {x.shape}")
print(f"  Time embedding shape: {time_emb.shape}")

with torch.no_grad():
    out = model(x, time_emb)

print(f"  Output shape: {out.shape}")
assert out.shape == (2, 1, 256, 256), f"Output shape mismatch: {out.shape}"
print("  ✓ PASSED")

# Test 2: Forward pass with CEM features
print("\n[Test 2] Forward pass with CEM features")
model_with_cem = SwinUNet(
    img_size=256,
    in_chans=3,
    out_chans=1,
    embed_dim=96,
    depths=[2, 2, 2, 2],
    num_heads=[3, 6, 12, 24],
    window_size=8,
    cem_in_channels=2,
    cem_enabled=True,
)

x = torch.randn(2, 3, 256, 256)
time_emb = torch.randn(2, 256)

# Simulate CEM features (normally computed by adapter)
cem_features = [
    torch.randn(2, 96, 64, 64),    # Stage 1
    torch.randn(2, 192, 32, 32),   # Stage 2
    torch.randn(2, 384, 16, 16),   # Stage 3
    torch.randn(2, 768, 8, 8),     # Stage 4
]

print(f"  Input shape: {x.shape}")
print(f"  CEM features: {[f.shape for f in cem_features]}")

with torch.no_grad():
    out = model_with_cem(x, time_emb, cem_features)

print(f"  Output shape: {out.shape}")
assert out.shape == (2, 1, 256, 256), f"Output shape mismatch: {out.shape}"
print("  ✓ PASSED")

# Test 3: Gradient flow
print("\n[Test 3] Gradient flow")
model = SwinUNet(
    img_size=256,
    in_chans=3,
    out_chans=1,
    embed_dim=96,
    depths=[2, 2, 2, 2],
    num_heads=[3, 6, 12, 24],
    window_size=8,
    cem_enabled=False,
)

x = torch.randn(2, 3, 256, 256, requires_grad=True)
time_emb = torch.randn(2, 256)

out = model(x, time_emb)
loss = out.sum()
loss.backward()

assert x.grad is not None, "Gradient not computed!"
assert x.grad.abs().sum() > 0, "Zero gradients!"
print(f"  Gradient sum: {x.grad.abs().sum().item():.4f}")
print("  ✓ PASSED")

# Test 4: Different embed dimensions
print("\n[Test 4] Different embed dimensions")
for embed_dim in [64, 96, 128]:
    model = SwinUNet(
        img_size=256,
        in_chans=3,
        out_chans=1,
        embed_dim=embed_dim,
        depths=[2, 2, 2, 2],
        num_heads=[2, 4, 8, 16] if embed_dim == 64 else [3, 6, 12, 24],
        window_size=8,
        cem_enabled=False,
    )
    
    x = torch.randn(2, 3, 256, 256)
    time_emb = torch.randn(2, 256)
    
    with torch.no_grad():
        out = model(x, time_emb)
    
    assert out.shape == (2, 1, 256, 256), f"Shape mismatch for embed_dim={embed_dim}"
    print(f"  embed_dim={embed_dim}: {out.shape} ✓")

print("  ✓ PASSED")

# Test 5: Different depth configurations
print("\n[Test 5] Different depth configurations")
depth_configs = [
    [2, 2, 2, 2],   # Shallow
    [2, 2, 6, 2],   # Standard Swin-T
    [2, 2, 18, 2],  # Deeper middle stage
]

for depths in depth_configs:
    model = SwinUNet(
        img_size=256,
        in_chans=3,
        out_chans=1,
        embed_dim=96,
        depths=depths,
        num_heads=[3, 6, 12, 24],
        window_size=8,
        cem_enabled=False,
    )
    
    x = torch.randn(2, 3, 256, 256)
    time_emb = torch.randn(2, 256)
    
    with torch.no_grad():
        out = model(x, time_emb)
    
    assert out.shape == (2, 1, 256, 256), f"Shape mismatch for depths={depths}"
    print(f"  depths={depths}: {out.shape} ✓")

print("  ✓ PASSED")

# Test 6: Different output channels
print("\n[Test 6] Different output channels")
for out_chans in [1, 2, 3]:
    model = SwinUNet(
        img_size=256,
        in_chans=3,
        out_chans=out_chans,
        embed_dim=96,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=8,
        cem_enabled=False,
    )
    
    x = torch.randn(2, 3, 256, 256)
    time_emb = torch.randn(2, 256)
    
    with torch.no_grad():
        out = model(x, time_emb)
    
    assert out.shape == (2, out_chans, 256, 256), f"Shape mismatch for out_chans={out_chans}"
    print(f"  out_chans={out_chans}: {out.shape} ✓")

print("  ✓ PASSED")

# Test 7: Time conditioning effect
print("\n[Test 7] Time conditioning affects output")
model = SwinUNet(
    img_size=256,
    in_chans=3,
    out_chans=1,
    embed_dim=96,
    depths=[2, 2, 2, 2],
    num_heads=[3, 6, 12, 24],
    window_size=8,
    cem_enabled=False,
)

x = torch.randn(2, 3, 256, 256)
time_emb_1 = torch.randn(2, 256)
time_emb_2 = torch.randn(2, 256)

with torch.no_grad():
    out_1 = model(x, time_emb_1)
    out_2 = model(x, time_emb_2)

# Different time embeddings should produce different outputs
difference = (out_1 - out_2).abs().mean().item()
print(f"  Mean absolute difference: {difference:.6f}")
assert difference > 1e-6, "Time conditioning has no effect!"
print("  ✓ PASSED")

# Test 8: CEM fusion effect
print("\n[Test 8] CEM fusion affects output")
model = SwinUNet(
    img_size=256,
    in_chans=3,
    out_chans=1,
    embed_dim=96,
    depths=[2, 2, 2, 2],
    num_heads=[3, 6, 12, 24],
    window_size=8,
    cem_enabled=True,
)

x = torch.randn(2, 3, 256, 256)
time_emb = torch.randn(2, 256)

cem_features_1 = [
    torch.randn(2, 96, 64, 64),
    torch.randn(2, 192, 32, 32),
    torch.randn(2, 384, 16, 16),
    torch.randn(2, 768, 8, 8),
]

cem_features_2 = [
    torch.randn(2, 96, 64, 64),
    torch.randn(2, 192, 32, 32),
    torch.randn(2, 384, 16, 16),
    torch.randn(2, 768, 8, 8),
]

with torch.no_grad():
    out_1 = model(x, time_emb, cem_features_1)
    out_2 = model(x, time_emb, cem_features_2)

difference = (out_1 - out_2).abs().mean().item()
print(f"  Mean absolute difference: {difference:.6f}")
assert difference > 1e-6, "CEM fusion has no effect!"
print("  ✓ PASSED")

# Test 9: Model parameter count
print("\n[Test 9] Model parameter count")
model_configs = [
    ("Small", 64, [2, 2, 2, 2], [2, 4, 8, 16]),
    ("Base", 96, [2, 2, 6, 2], [3, 6, 12, 24]),
    ("Large", 128, [2, 2, 18, 2], [4, 8, 16, 32]),
]

for name, embed_dim, depths, num_heads in model_configs:
    model = SwinUNet(
        img_size=256,
        in_chans=3,
        out_chans=1,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=8,
        cem_enabled=False,
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  {name} (embed={embed_dim}, depths={depths}): {num_params:,} params")

print("  ✓ PASSED")

# Test 10: Batch size flexibility
print("\n[Test 10] Different batch sizes")
model = SwinUNet(
    img_size=256,
    in_chans=3,
    out_chans=1,
    embed_dim=96,
    depths=[2, 2, 2, 2],
    num_heads=[3, 6, 12, 24],
    window_size=8,
    cem_enabled=False,
)

for batch_size in [1, 2, 4, 8]:
    x = torch.randn(batch_size, 3, 256, 256)
    time_emb = torch.randn(batch_size, 256)
    
    with torch.no_grad():
        out = model(x, time_emb)
    
    assert out.shape == (batch_size, 1, 256, 256), f"Shape mismatch for batch={batch_size}"
    print(f"  Batch size {batch_size}: {out.shape} ✓")

print("  ✓ PASSED")

print("\n" + "=" * 60)
print("ALL SWINUNET TESTS PASSED! ✓")
print("=" * 60)

