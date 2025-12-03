"""
Test script for DiffSwinTr configuration files.
Run with: python -m tests.test_configs
"""
import torch
from omegaconf import OmegaConf
from src.models.model_factory import build_model

print("=" * 60)
print("Testing DiffSwinTr Configuration Files")
print("=" * 60)

# Test 1: Load diffswintr_b.yaml
print("\n[Test 1] Load diffswintr_b.yaml (Base)")
cfg_b = OmegaConf.load('configs/model/diffswintr_b.yaml')

# Add dataset config for interpolation
cfg_b_full = OmegaConf.create({
    'dataset': {'num_modalities': 2},
    'model': cfg_b
})
OmegaConf.resolve(cfg_b_full)

print(f"  Architecture: {cfg_b_full.model.architecture}")
print(f"  Embed dim: {cfg_b_full.model.embed_dim}")
print(f"  Depths: {cfg_b_full.model.depths}")
print(f"  Num heads: {cfg_b_full.model.num_heads}")
print(f"  Image channels (interpolated): {cfg_b_full.model.image_channels}")

assert cfg_b_full.model.architecture == "diffswintr"
assert cfg_b_full.model.embed_dim == 96
assert cfg_b_full.model.image_channels == 2
print("  ✓ PASSED")

# Test 2: Build model from diffswintr_b.yaml
print("\n[Test 2] Build model from diffswintr_b.yaml")
model_b = build_model(cfg_b_full)
print(f"  Model type: {type(model_b).__name__}")

num_params_b = sum(p.numel() for p in model_b.parameters())
print(f"  Parameters: {num_params_b:,}")

# Test forward pass
x = torch.randn(2, 1, 256, 256)
t = torch.randint(0, 1000, (2,))
img = torch.randn(2, 2, 256, 256)

with torch.no_grad():
    out = model_b(x, t, img)

assert out.shape == (2, 1, 256, 256)
print(f"  Output shape: {out.shape}")
print("  ✓ PASSED")

# Test 3: Load diffswintr_s.yaml (with defaults)
print("\n[Test 3] Load diffswintr_s.yaml (Small)")
# Load with Hydra's compose API is ideal, but we'll manually merge
cfg_s = OmegaConf.load('configs/model/diffswintr_s.yaml')
cfg_b_base = OmegaConf.load('configs/model/diffswintr_b.yaml')

# Merge: base first, then small overrides
cfg_s_merged = OmegaConf.merge(cfg_b_base, cfg_s)

cfg_s_full = OmegaConf.create({
    'dataset': {'num_modalities': 2},
    'model': cfg_s_merged
})
OmegaConf.resolve(cfg_s_full)

print(f"  Architecture: {cfg_s_full.model.architecture}")
print(f"  Embed dim: {cfg_s_full.model.embed_dim}")
print(f"  Depths: {cfg_s_full.model.depths}")
print(f"  Num heads: {cfg_s_full.model.num_heads}")

assert cfg_s_full.model.embed_dim == 64, f"Expected 64, got {cfg_s_full.model.embed_dim}"
assert cfg_s_full.model.depths == [2, 2, 2, 2]
assert cfg_s_full.model.num_heads == [2, 4, 8, 16]
print("  ✓ PASSED")

# Test 4: Build model from diffswintr_s.yaml
print("\n[Test 4] Build model from diffswintr_s.yaml")
model_s = build_model(cfg_s_full)

num_params_s = sum(p.numel() for p in model_s.parameters())
print(f"  Parameters: {num_params_s:,}")

with torch.no_grad():
    out = model_s(x, t, img)

assert out.shape == (2, 1, 256, 256)
print(f"  Output shape: {out.shape}")
print("  ✓ PASSED")

# Test 5: Load diffswintr_l.yaml (with defaults)
print("\n[Test 5] Load diffswintr_l.yaml (Large)")
cfg_l = OmegaConf.load('configs/model/diffswintr_l.yaml')
cfg_l_merged = OmegaConf.merge(cfg_b_base, cfg_l)

cfg_l_full = OmegaConf.create({
    'dataset': {'num_modalities': 2},
    'model': cfg_l_merged
})
OmegaConf.resolve(cfg_l_full)

print(f"  Architecture: {cfg_l_full.model.architecture}")
print(f"  Embed dim: {cfg_l_full.model.embed_dim}")
print(f"  Depths: {cfg_l_full.model.depths}")
print(f"  Num heads: {cfg_l_full.model.num_heads}")

assert cfg_l_full.model.embed_dim == 128
assert cfg_l_full.model.depths == [2, 2, 18, 2]
assert cfg_l_full.model.num_heads == [4, 8, 16, 32]
print("  ✓ PASSED")

# Test 6: Build model from diffswintr_l.yaml
print("\n[Test 6] Build model from diffswintr_l.yaml")
model_l = build_model(cfg_l_full)

num_params_l = sum(p.numel() for p in model_l.parameters())
print(f"  Parameters: {num_params_l:,}")

with torch.no_grad():
    out = model_l(x, t, img)

assert out.shape == (2, 1, 256, 256)
print(f"  Output shape: {out.shape}")
print("  ✓ PASSED")

# Test 7: Verify parameter counts are different
print("\n[Test 7] Verify model size progression")
print(f"  Small:  {num_params_s:>12,} params")
print(f"  Base:   {num_params_b:>12,} params")
print(f"  Large:  {num_params_l:>12,} params")

assert num_params_s < num_params_b < num_params_l, "Size progression incorrect!"
print("  ✓ Size progression: Small < Base < Large")
print("  ✓ PASSED")

# Test 8: Verify all required fields present
print("\n[Test 8] Verify all required config fields")
required_fields = [
    'architecture', 'image_size', 'patch_size', 'embed_dim', 'window_size',
    'depths', 'num_heads', 'mlp_ratio', 'time_embed_dim', 
    'mask_channels', 'image_channels', 'out_channels',
    'cem_enabled', 'cem_fusion_mode'
]

for field in required_fields:
    assert hasattr(cfg_b_full.model, field), f"Missing field: {field}"
    print(f"  ✓ {field}: {getattr(cfg_b_full.model, field)}")

print("  ✓ PASSED")

print("\n" + "=" * 60)
print("ALL CONFIG TESTS PASSED! ✓")
print("=" * 60)
print("\nConfiguration files are ready for use!")
print("  - configs/model/diffswintr_s.yaml (Small)")
print("  - configs/model/diffswintr_b.yaml (Base)")
print("  - configs/model/diffswintr_l.yaml (Large)")

