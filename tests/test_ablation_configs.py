#!/usr/bin/env python3
"""
Test ablation: Multiple optimizer/scheduler combinations.

Task 5.4: Tests systematic ablation across different configurations
to ensure all optimizer×scheduler combinations work correctly.

Usage:
    python tests/test_ablation_configs.py
"""

import subprocess
import sys
from datetime import datetime


def run_training_test(optimizer, scheduler, max_steps=100):
    """
    Run a short training test with given optimizer/scheduler config.
    
    Args:
        optimizer: Optimizer config name (e.g., 'adamw_1e4lr_wd00')
        scheduler: Scheduler config name (e.g., 'warmup_cosine_10pct')
        max_steps: Number of training steps (default: 100)
    
    Returns:
        True if successful, False otherwise
    """
    cmd = [
        'python', 'main.py',
        f'optimizer={optimizer}',
        f'scheduler={scheduler}',
        f'training.max_steps={max_steps}',
        '--config-name=local_openai_ddim250'
    ]
    
    print(f"\n{'='*70}")
    print(f"Testing: {optimizer} + {scheduler}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode == 0:
            print(f"✅ PASSED: {optimizer} + {scheduler}")
            return True
        else:
            print(f"❌ FAILED: {optimizer} + {scheduler}")
            print(f"\nSTDOUT:\n{result.stdout[-2000:]}")  # Last 2000 chars
            print(f"\nSTDERR:\n{result.stderr[-2000:]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT: {optimizer} + {scheduler} (exceeded 10 minutes)")
        return False
    except Exception as e:
        print(f"❌ ERROR: {optimizer} + {scheduler}")
        print(f"Exception: {e}")
        return False


def main():
    """Run ablation test matrix."""
    print("="*70)
    print("Phase 5 - Task 5.4: Ablation Config Testing")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Define test matrix: (optimizer, scheduler, purpose)
    test_configs = [
        ('adamw_1e4lr_wd00', 'warmup_cosine_10pct', 'Recommended (DiffSwinTr-style)'),
        ('adamw_2e4lr_wd00', 'warmup_cosine_10pct', 'Higher LR'),
        ('adamw_1e4lr_wd00', 'cosine', 'No warmup (ablation)'),
        ('adam_2e4lr', 'reduce_lr_075f_2p_1e4t_1c_1Ki', 'Legacy ReduceLR'),
    ]
    
    print(f"\nTotal configurations to test: {len(test_configs)}")
    print("\nTest Matrix:")
    for i, (opt, sched, purpose) in enumerate(test_configs, 1):
        print(f"  {i}. {opt:<25} + {sched:<25} ({purpose})")
    
    # Run tests
    results = []
    for optimizer, scheduler, purpose in test_configs:
        success = run_training_test(optimizer, scheduler, max_steps=100)
        results.append((optimizer, scheduler, purpose, success))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, _, _, success in results if success)
    failed = len(results) - passed
    
    print(f"\nResults: {passed}/{len(results)} passed, {failed}/{len(results)} failed")
    print("\nDetailed Results:")
    for i, (opt, sched, purpose, success) in enumerate(results, 1):
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {i}. {status} | {opt:<25} + {sched:<25} ({purpose})")
    
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    if failed > 0:
        print(f"\n❌ {failed} test(s) failed!")
        sys.exit(1)
    else:
        print("\n✅ All ablation tests passed!")
        sys.exit(0)


if __name__ == '__main__':
    main()

