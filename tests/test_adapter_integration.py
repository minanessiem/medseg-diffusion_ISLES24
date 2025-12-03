"""
Integration tests for DiffSwinTr adapter and model factory.
Run with: python -m tests.test_adapter_integration
"""
import torch
from omegaconf import OmegaConf
from src.models.model_factory import build_model
from src.models.DiffSwinTr import DiffSwinTrAdapter

print("=" * 60)
print("Testing DiffSwinTr Adapter and Integration")
print("=" * 60)

# Test 1: Direct adapter instantiation
print("\n[Test 1] Direct adapter instantiation")
cfg = OmegaConf.create({
    'model': {
        'architecture': 'diffswintr',
        'image_size': 256,
        'patch_size': 4,
        'embed_dim': 96,
        'window_size': 8,
        'depths': [2, 2, 2, 2],
        'num_heads': [3, 6, 12, 24],
        'mlp_ratio': 4.0,
        'time_embed_dim': 256,
        'mask_channels': 1,
        'image_channels': 2,
        'out_channels': 1,
        'cem_enabled': True,
        'cem_fusion_mode': 'add',
        'drop_rate': 0.0,
        'attn_drop_rate': 0.0,
        'drop_path_rate': 0.1,
    }
})

model = DiffSwinTrAdapter(cfg)
print(f"  Model type: {type(model).__name__}")
print(f"  Image size: {model.image_size}")
print(f"  Mask channels: {model.mask_channels}")
print(f"  Image channels: {model.image_channels}")
print(f"  Produces calibration: {model.produces_calibration}")
print("  ✓ PASSED")

# Test 2: Pipeline interface
print("\n[Test 2] Pipeline interface (forward pass)")
x = torch.randn(2, 1, 256, 256)  # Noisy mask
t = torch.randint(0, 1000, (2,))  # Timesteps
img = torch.randn(2, 2, 256, 256)  # Conditioning image

print(f"  Input x shape: {x.shape}")
print(f"  Timesteps shape: {t.shape}")
print(f"  Conditioning image shape: {img.shape}")

with torch.no_grad():
    out = model(x, t, img)

print(f"  Output shape: {out.shape}")
assert out.shape == (2, 1, 256, 256), f"Output shape mismatch: {out.shape}"
print("  ✓ PASSED")

# Test 3: Model factory build
print("\n[Test 3] Build via model factory")
model_from_factory = build_model(cfg)
print(f"  Model type: {type(model_from_factory).__name__}")
assert isinstance(model_from_factory, DiffSwinTrAdapter), "Wrong model type!"
print("  ✓ PASSED")

# Test 4: Config with string depths/num_heads
print("\n[Test 4] Config with comma-separated strings")
cfg_str = OmegaConf.create({
    'model': {
        'architecture': 'diffswintr',
        'image_size': 256,
        'patch_size': 4,
        'embed_dim': 96,
        'window_size': 8,
        'depths': '2,2,2,2',  # String format
        'num_heads': '3,6,12,24',  # String format
        'mlp_ratio': 4.0,
        'time_embed_dim': 256,
        'mask_channels': 1,
        'image_channels': 2,
        'out_channels': 1,
        'cem_enabled': True,
    }
})

model_str = DiffSwinTrAdapter(cfg_str)
x = torch.randn(2, 1, 256, 256)
t = torch.randint(0, 1000, (2,))
img = torch.randn(2, 2, 256, 256)

with torch.no_grad():
    out = model_str(x, t, img)

assert out.shape == (2, 1, 256, 256), "String config failed!"
print("  ✓ PASSED")

# Test 5: CEM disabled
print("\n[Test 5] CEM disabled configuration")
cfg_no_cem = OmegaConf.create({
    'model': {
        'architecture': 'diffswintr',
        'image_size': 256,
        'patch_size': 4,
        'embed_dim': 96,
        'window_size': 8,
        'depths': [2, 2, 2, 2],
        'num_heads': [3, 6, 12, 24],
        'mlp_ratio': 4.0,
        'time_embed_dim': 256,
        'mask_channels': 1,
        'image_channels': 2,
        'out_channels': 1,
        'cem_enabled': False,  # Disabled
    }
})

model_no_cem = DiffSwinTrAdapter(cfg_no_cem)
assert model_no_cem.cem is None, "CEM should be None!"

x = torch.randn(2, 1, 256, 256)
t = torch.randint(0, 1000, (2,))
img = torch.randn(2, 2, 256, 256)

with torch.no_grad():
    out = model_no_cem(x, t, img)

assert out.shape == (2, 1, 256, 256), "No CEM forward failed!"
print("  ✓ PASSED")

# Test 6: Gradient flow through adapter
print("\n[Test 6] Gradient flow through adapter")
model = DiffSwinTrAdapter(cfg)
x = torch.randn(2, 1, 256, 256, requires_grad=True)
t = torch.randint(0, 1000, (2,))
img = torch.randn(2, 2, 256, 256)

out = model(x, t, img)
loss = out.sum()
loss.backward()

assert x.grad is not None, "Gradient not computed!"
assert x.grad.abs().sum() > 0, "Zero gradients!"
print(f"  Gradient sum: {x.grad.abs().sum().item():.4f}")
print("  ✓ PASSED")

# Test 7: Adapter attributes
print("\n[Test 7] Adapter attributes match config")
assert model.image_size == 256, "image_size mismatch!"
assert model.mask_channels == 1, "mask_channels mismatch!"
assert model.image_channels == 2, "image_channels mismatch!"
assert model.output_channels == 1, "output_channels mismatch!"
assert model.produces_calibration == False, "produces_calibration should be False!"
print("  All attributes match config ✓")
print("  ✓ PASSED")

# Test 8: Different model sizes
print("\n[Test 8] Different model sizes")
size_configs = [
    ("Small", 64, [2, 2, 2, 2], [2, 4, 8, 16]),
    ("Base", 96, [2, 2, 6, 2], [3, 6, 12, 24]),
    ("Large", 128, [2, 2, 18, 2], [4, 8, 16, 32]),
]

for name, embed_dim, depths, num_heads in size_configs:
    cfg_size = OmegaConf.create({
        'model': {
            'architecture': 'diffswintr',
            'image_size': 256,
            'patch_size': 4,
            'embed_dim': embed_dim,
            'window_size': 8,
            'depths': depths,
            'num_heads': num_heads,
            'mlp_ratio': 4.0,
            'time_embed_dim': 256,
            'mask_channels': 1,
            'image_channels': 2,
            'out_channels': 1,
            'cem_enabled': True,
        }
    })
    
    model_size = build_model(cfg_size)
    
    x = torch.randn(2, 1, 256, 256)
    t = torch.randint(0, 1000, (2,))
    img = torch.randn(2, 2, 256, 256)
    
    with torch.no_grad():
        out = model_size(x, t, img)
    
    num_params = sum(p.numel() for p in model_size.parameters())
    print(f"  {name}: {out.shape}, {num_params:,} params ✓")

print("  ✓ PASSED")

# Test 9: Test with dataset config interpolation
print("\n[Test 9] Config with dataset interpolation")
cfg_with_dataset = OmegaConf.create({
    'dataset': {
        'num_modalities': 2,
    },
    'model': {
        'architecture': 'diffswintr',
        'image_size': 256,
        'patch_size': 4,
        'embed_dim': 96,
        'window_size': 8,
        'depths': [2, 2, 2, 2],
        'num_heads': [3, 6, 12, 24],
        'mlp_ratio': 4.0,
        'time_embed_dim': 256,
        'mask_channels': 1,
        'image_channels': '${dataset.num_modalities}',  # Interpolation
        'out_channels': 1,
        'cem_enabled': True,
    }
})

# Resolve interpolations
OmegaConf.resolve(cfg_with_dataset)

model_interp = build_model(cfg_with_dataset)
assert model_interp.image_channels == 2, "Interpolation failed!"
print(f"  Interpolated image_channels: {model_interp.image_channels}")
print("  ✓ PASSED")

print("\n" + "=" * 60)
print("ALL ADAPTER INTEGRATION TESTS PASSED! ✓")
print("=" * 60)

