"""
Test run name generation with new optimizer/scheduler separation.

Tests Phase 4 implementation:
- format_learning_rate()
- generate_optimizer_string()
- generate_scheduler_string()
- generate_run_name() full integration
"""

from omegaconf import OmegaConf
from src.utils.run_name import (
    format_learning_rate,
    generate_optimizer_string,
    generate_scheduler_string,
    generate_run_name
)


def test_format_learning_rate():
    """Test LR formatting."""
    print("\n[Test 1] format_learning_rate()")
    print("-" * 70)
    test_lrs = [1e-4, 2e-4, 5e-5, 1e-3]
    for lr in test_lrs:
        result = format_learning_rate(lr)
        print(f"  {lr:.0e} → '{result}'")
    
    # Assertions
    assert format_learning_rate(1e-4) == "1e4"
    assert format_learning_rate(2e-4) == "2e4"
    assert format_learning_rate(5e-5) == "5e5"
    print("✓ All assertions passed")


def test_generate_optimizer_string():
    """Test optimizer string generation."""
    print("\n[Test 2] generate_optimizer_string()")
    print("-" * 70)
    
    opt_configs = [
        {'optimizer_class': 'adamw', 'learning_rate': 1e-4, 'weight_decay': 0.0},
        {'optimizer_class': 'adamw', 'learning_rate': 1e-4, 'weight_decay': 0.01},
        {'optimizer_class': 'adamw', 'learning_rate': 2e-4, 'weight_decay': 0.0},
        {'optimizer_class': 'adam', 'learning_rate': 2e-4, 'weight_decay': 0.0},
    ]
    
    for opt_cfg in opt_configs:
        result = generate_optimizer_string(opt_cfg)
        print(f"  {opt_cfg['optimizer_class']}, lr={opt_cfg['learning_rate']:.0e}, wd={opt_cfg['weight_decay']} → '{result}'")
    
    # Assertions
    assert generate_optimizer_string(opt_configs[0]) == "adamw1e4_wd00"
    assert generate_optimizer_string(opt_configs[1]) == "adamw1e4_wd01"
    assert generate_optimizer_string(opt_configs[2]) == "adamw2e4_wd00"
    assert generate_optimizer_string(opt_configs[3]) == "adam2e4_wd00"
    print("✓ All assertions passed")


def test_generate_scheduler_string():
    """Test scheduler string generation."""
    print("\n[Test 3] generate_scheduler_string()")
    print("-" * 70)
    
    sched_configs = [
        {'scheduler_type': 'warmup_cosine', 'warmup_fraction': 0.1},
        {'scheduler_type': 'warmup_cosine', 'warmup_fraction': 0.05},
        {'scheduler_type': 'warmup_constant', 'warmup_fraction': 0.1},
        {'scheduler_type': 'cosine'},
        {'scheduler_type': 'reduce_lr', 'factor': 0.75, 'patience': 2},
        {'scheduler_type': 'constant'},
    ]
    
    for sched_cfg in sched_configs:
        result = generate_scheduler_string(sched_cfg)
        sched_type = sched_cfg['scheduler_type']
        extras = ', '.join([f"{k}={v}" for k, v in sched_cfg.items() if k != 'scheduler_type'])
        print(f"  {sched_type}{(' (' + extras + ')') if extras else ''} → '{result}'")
    
    # Assertions
    assert generate_scheduler_string(sched_configs[0]) == "wcos10"
    assert generate_scheduler_string(sched_configs[1]) == "wcos5"
    assert generate_scheduler_string(sched_configs[2]) == "wcon10"
    assert generate_scheduler_string(sched_configs[3]) == "cos"
    assert generate_scheduler_string(sched_configs[4]) == "rlr75f_2p"
    assert generate_scheduler_string(sched_configs[5]) == "const"
    print("✓ All assertions passed")


def test_full_run_name_generation():
    """Test full run name generation with all components."""
    print("\n[Test 4] generate_run_name() - Full integration")
    print("-" * 70)
    
    # Create minimal config
    cfg_dict = {
        'model': {
            'architecture': 'medsegdiff',
            'image_size': 256,
            'num_layers': 4,
            'first_conv_channels': 16,
            'att_heads': 6,
            'att_head_dim': 4,
            'time_embedding_dim': 128,
            'bottleneck_transformer_layers': 1
        },
        'dataset': {
            'train_batch_size': 4
        },
        'optimizer': {
            'optimizer_class': 'adamw',
            'learning_rate': 1e-4,
            'weight_decay': 0.0
        },
        'scheduler': {
            'scheduler_type': 'warmup_cosine',
            'warmup_fraction': 0.1
        },
        'training': {
            'max_steps': 100000
        },
        'loss': {
            'loss_type': 'MSE',
            'auxiliary_losses': {'enabled': False}
        },
        'diffusion': {
            'type': 'OpenAI_DDPM',
            'sampling_mode': 'ddim',
            'timesteps': 1000,
            'noise_schedule': 'cosine',
            'timestep_respacing': '250'
        }
    }
    
    cfg = OmegaConf.create(cfg_dict)
    run_name = generate_run_name(cfg, timestamp='2025-11-19_12-00-00')
    
    print(f"Generated run name:")
    print(f"  {run_name}")
    print()
    print("Breakdown:")
    print(f"  Model:     medsegdiff_256_4l_16c_6x4a_128t_1btl")
    print(f"  Batch:     b4")
    print(f"  Optimizer: adamw1e4_wd00")
    print(f"  Scheduler: wcos10")
    print(f"  Steps:     s100K")
    print(f"  Loss:      lMSE")
    print(f"  Diffusion: oai_ddim_ds1000_nzcosine_tr250")
    print(f"  Timestamp: 2025-11-19_12-00-00")
    
    # Assertions
    assert "adamw1e4_wd00" in run_name
    assert "wcos10" in run_name
    assert "s100K" in run_name
    print("✓ All assertions passed")


def test_various_combinations():
    """Test various optimizer/scheduler combinations."""
    print("\n[Test 5] Various optimizer/scheduler combinations")
    print("-" * 70)
    
    # Create base config
    cfg_dict = {
        'model': {'architecture': 'medsegdiff', 'image_size': 256, 'num_layers': 4,
                  'first_conv_channels': 16, 'att_heads': 6, 'att_head_dim': 4,
                  'time_embedding_dim': 128, 'bottleneck_transformer_layers': 1},
        'dataset': {'train_batch_size': 4},
        'optimizer': {'optimizer_class': 'adamw', 'learning_rate': 1e-4, 'weight_decay': 0.0},
        'scheduler': {'scheduler_type': 'warmup_cosine', 'warmup_fraction': 0.1},
        'training': {'max_steps': 100000},
        'loss': {'loss_type': 'MSE', 'auxiliary_losses': {'enabled': False}},
        'diffusion': {'type': 'OpenAI_DDPM', 'sampling_mode': 'ddim', 'timesteps': 1000,
                     'noise_schedule': 'cosine', 'timestep_respacing': '250'}
    }
    cfg = OmegaConf.create(cfg_dict)
    
    combinations = [
        ('adamw_2e4lr_wd00', 'warmup_cosine_10pct'),
        ('adamw_1e4lr_wd01', 'cosine'),
        ('adam_2e4lr', 'reduce_lr_075f_2p'),
        ('adamw_1e4lr_wd00', 'constant'),
    ]
    
    for opt_name, sched_name in combinations:
        # Update config
        if 'adamw' in opt_name:
            cfg.optimizer.optimizer_class = 'adamw'
        else:
            cfg.optimizer.optimizer_class = 'adam'
        
        if '2e4' in opt_name:
            cfg.optimizer.learning_rate = 2e-4
        else:
            cfg.optimizer.learning_rate = 1e-4
        
        if 'wd01' in opt_name:
            cfg.optimizer.weight_decay = 0.01
        else:
            cfg.optimizer.weight_decay = 0.0
        
        if 'warmup_cosine' in sched_name:
            cfg.scheduler.scheduler_type = 'warmup_cosine'
            if '5pct' in sched_name:
                cfg.scheduler.warmup_fraction = 0.05
            else:
                cfg.scheduler.warmup_fraction = 0.1
        elif 'cosine' in sched_name and 'warmup' not in sched_name:
            cfg.scheduler.scheduler_type = 'cosine'
        elif 'reduce_lr' in sched_name:
            cfg.scheduler.scheduler_type = 'reduce_lr'
            cfg.scheduler.factor = 0.75
            cfg.scheduler.patience = 2
        elif 'constant' in sched_name:
            cfg.scheduler.scheduler_type = 'constant'
        
        opt_str = generate_optimizer_string(cfg.optimizer)
        sched_str = generate_scheduler_string(cfg.scheduler)
        
        print(f"  {opt_name:<25} + {sched_name:<25} → {opt_str}_{sched_str}")
    
    print("✓ All combinations tested")


def run_all_tests():
    """Run all run name generation tests."""
    print("=" * 70)
    print("Phase 4 Validation: Run Name Generation")
    print("=" * 70)
    
    test_format_learning_rate()
    test_generate_optimizer_string()
    test_generate_scheduler_string()
    test_full_run_name_generation()
    test_various_combinations()
    
    print("\n" + "=" * 70)
    print("✅ Phase 4 Validation: ALL TESTS PASSED!")
    print("=" * 70)


if __name__ == '__main__':
    run_all_tests()

