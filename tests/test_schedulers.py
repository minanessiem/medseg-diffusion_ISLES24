"""
Test custom learning rate schedulers.

Tests Phase 1 implementation:
- WarmupCosineSchedule
- WarmupConstantSchedule
- Factory function imports
"""
import torch
import torch.nn as nn
from src.training.schedulers import WarmupCosineSchedule, WarmupConstantSchedule


def test_warmup_cosine_schedule():
    """Test WarmupCosineSchedule behavior."""
    print("\n[Test 1] WarmupCosineSchedule")
    print("-" * 60)
    
    # Dummy model
    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=10, total_steps=100)
    
    lrs = []
    for step in range(100):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    
    print(f"✓ Scheduler instantiated successfully")
    print(f"✓ LR range: {min(lrs):.6e} → {max(lrs):.6e}")
    print(f"  Step 0 LR:   {lrs[0]:.6e} (should be 0)")
    print(f"  Step 10 LR:  {lrs[10]:.6e} (should be 1e-4)")
    print(f"  Step 50 LR:  {lrs[50]:.6e}")
    print(f"  Step 99 LR:  {lrs[99]:.6e} (should be near 0)")
    
    # Assertions
    assert lrs[0] == 0.0, f"❌ LR should start at 0, got {lrs[0]}"
    assert abs(lrs[10] - 1e-4) < 1e-10, f"❌ LR should reach base_lr (1e-4) after warmup, got {lrs[10]}"
    assert lrs[-1] < 1e-5, f"❌ LR should decay to near 0, got {lrs[-1]}"
    print("✓ All assertions passed")


def test_warmup_constant_schedule():
    """Test WarmupConstantSchedule behavior."""
    print("\n[Test 2] WarmupConstantSchedule")
    print("-" * 60)
    
    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    scheduler = WarmupConstantSchedule(optimizer, warmup_steps=10)
    
    lrs = []
    for step in range(50):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    
    print(f"✓ Scheduler instantiated successfully")
    print(f"✓ LR range: {min(lrs):.6e} → {max(lrs):.6e}")
    print(f"  Step 0 LR:   {lrs[0]:.6e} (should be 0)")
    print(f"  Step 10 LR:  {lrs[10]:.6e} (should be 2e-4)")
    print(f"  Step 25 LR:  {lrs[25]:.6e} (should be 2e-4)")
    print(f"  Step 49 LR:  {lrs[49]:.6e} (should be 2e-4)")
    
    # Assertions
    assert lrs[0] == 0.0, f"❌ LR should start at 0, got {lrs[0]}"
    assert abs(lrs[10] - 2e-4) < 1e-10, f"❌ LR should reach base_lr (2e-4), got {lrs[10]}"
    assert abs(lrs[-1] - 2e-4) < 1e-10, f"❌ LR should stay constant at 2e-4, got {lrs[-1]}"
    print("✓ All assertions passed")


def test_factory_imports():
    """Test that factory functions can be imported."""
    print("\n[Test 3] Factory Function Imports")
    print("-" * 60)
    
    try:
        from src.training.optimizer_factory import build_optimizer, build_scheduler
        print("✓ build_optimizer imported successfully")
        print("✓ build_scheduler imported successfully")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        raise


def run_all_tests():
    """Run all scheduler tests."""
    print("=" * 60)
    print("Phase 1 Validation: Custom Schedulers")
    print("=" * 60)
    
    test_warmup_cosine_schedule()
    test_warmup_constant_schedule()
    test_factory_imports()
    
    print("\n" + "=" * 60)
    print("✅ Phase 1 Validation: ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()

