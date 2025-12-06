#!/usr/bin/env python3
"""
Smoke test for validation memory configuration.

This script tests whether a given configuration can run validation
without OOM errors. It builds the model, runs one validation batch,
and reports memory usage per GPU.

Usage:
    python scripts/test_validation_memory.py --config-name <config>
    
    # Example:
    python scripts/test_validation_memory.py --config-name cluster_ddim250_multitask
    
Exit codes:
    0: Success - validation completed without errors
    1: Failure - OOM or other error occurred
"""

import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import torch


@hydra.main(version_base=None, config_path="../configs", config_name="local")
def main(cfg: DictConfig):
    """Run validation smoke test."""
    
    print("=" * 60)
    print("VALIDATION MEMORY SMOKE TEST")
    print("=" * 60)
    
    # Setup
    from src.utils.train_utils import (
        setup_device,
        setup_config_aliases,
        build_model_and_diffusion,
        _parse_multi_gpu_flag,
    )
    from src.data.loaders import get_dataloaders
    from src.metrics.metrics import get_metric
    
    # Apply config aliases
    cfg = setup_config_aliases(cfg)
    device = setup_device(cfg)
    
    # Parse GPU config
    gpu_ids = _parse_multi_gpu_flag(cfg.environment.training.multi_gpu)
    if gpu_ids:
        print(f"Multi-GPU config: {gpu_ids}")
    else:
        gpu_ids = [0]  # Default to GPU 0
        print("Single-GPU config")
    
    # Check validation config
    multi_gpu_val = cfg.validation.get('multi_gpu_validation', False)
    val_batch_size = cfg.validation.val_batch_size
    print(f"Validation batch size: {val_batch_size}")
    print(f"Multi-GPU validation enabled: {multi_gpu_val}")
    
    # Build model
    print("\nBuilding model and diffusion...")
    diffusion = build_model_and_diffusion(cfg, device)
    print("Model built successfully")
    
    # Get validation dataloader
    print("\nLoading validation data...")
    dataloaders = get_dataloaders(cfg)
    val_dl = dataloaders['val']
    print(f"Validation samples: {len(val_dl.dataset)}")
    
    # Reset memory stats
    for gpu_id in gpu_ids:
        torch.cuda.reset_peak_memory_stats(gpu_id)
    
    # Run one validation batch
    print("\n" + "-" * 60)
    print("Running validation smoke test (1 batch)...")
    print("-" * 60)
    
    try:
        # Get one batch
        img, true_mask, *_ = next(iter(val_dl))
        print(f"Batch shape: img={img.shape}, mask={true_mask.shape}")
        
        if multi_gpu_val and gpu_ids and len(gpu_ids) > 1:
            # Test multi-GPU validation
            from src.utils.valid_utils import (
                get_unwrapped_model,
                create_model_copies,
                sample_parallel,
                cleanup_model_copies,
                log_gpu_memory,
            )
            
            print(f"\nTesting multi-GPU sampling across {len(gpu_ids)} GPUs...")
            
            base_model = get_unwrapped_model(diffusion.model)
            print("Creating model copies...")
            models = create_model_copies(base_model, gpu_ids)
            print("Model copies created")
            
            log_gpu_memory(gpu_ids, "after model copy")
            
            try:
                print("\nRunning parallel sampling...")
                pred_mask = sample_parallel(
                    diffusion_adapter=diffusion,
                    conditioned_images=img,
                    models=models,
                    gpu_ids=gpu_ids,
                )
                print(f"Sampling successful! Output shape: {pred_mask.shape}")
            finally:
                print("\nCleaning up model copies...")
                cleanup_model_copies(models)
            
            log_gpu_memory(gpu_ids, "peak")
            
        else:
            # Test single-GPU validation
            print("\nTesting single-GPU sampling...")
            img = img.to(device)
            pred_mask = diffusion.sample(img, disable_tqdm=False)
            print(f"Sampling successful! Output shape: {pred_mask.shape}")
            
            # Log memory
            peak_mb = torch.cuda.max_memory_allocated(0) / 1024**2
            print(f"\nGPU 0 peak memory: {peak_mb:.1f} MB")
        
        print("\n" + "=" * 60)
        print("✓ SMOKE TEST PASSED")
        print("=" * 60)
        return 0
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n✗ OUT OF MEMORY ERROR:")
        print(f"  {e}")
        print("\nSuggestions:")
        print("  1. Reduce validation.val_batch_size")
        print("  2. Enable validation.multi_gpu_validation if using multi-GPU")
        print("  3. Use a smaller model")
        print("\n" + "=" * 60)
        print("✗ SMOKE TEST FAILED (OOM)")
        print("=" * 60)
        sys.exit(1)
        
    except Exception as e:
        print(f"\n✗ ERROR:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 60)
        print("✗ SMOKE TEST FAILED")
        print("=" * 60)
        sys.exit(1)


if __name__ == '__main__':
    main()

