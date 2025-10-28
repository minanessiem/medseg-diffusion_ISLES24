import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.general import device_grad_decorator
from src.evaluation.evaluator import sample_and_visualize  # For visualization in train_and_evaluate
from src.utils.logger import Logger
from src.utils.train_utils import calc_grad_norm, calc_param_norm
import torch
from torch import no_grad
from src.models.architectures.unet_util import unnormalize_to_zero_to_one, normalize_to_neg_one_to_one
from src.metrics.metrics import get_metric

import gc
from omegaconf import OmegaConf

def get_optimizer_and_scheduler(cfg, model):
    """
    Create optimizer and learning rate scheduler based on configuration.

    Args:
        cfg (DictConfig): Hydra configuration object.
        model (nn.Module): The model to optimize.

    Returns:
        optimizer, scheduler
    """
    OmegaConf.set_struct(cfg, False)
    # Temporary aliases for config transition
    cfg.training.learning_rate = cfg.optimizer.learning_rate
    cfg.training.scheduler_type = cfg.optimizer.scheduler_type
    cfg.training.reduce_lr_factor = cfg.optimizer.reduce_lr_factor
    cfg.training.reduce_lr_patience = cfg.optimizer.reduce_lr_patience
    cfg.training.reduce_lr_threshold = cfg.optimizer.reduce_lr_threshold
    cfg.training.reduce_lr_cooldown = cfg.optimizer.reduce_lr_cooldown
    cfg.training.max_steps = cfg.training.max_steps  # Retained in training
    OmegaConf.set_struct(cfg, True)

    optimizer = Adam(model.parameters(), lr=cfg.training.learning_rate)
    
    if cfg.training.scheduler_type == 'reduce_lr':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=cfg.training.reduce_lr_factor,
            patience=cfg.training.reduce_lr_patience,
            threshold=cfg.training.reduce_lr_threshold,
            cooldown=cfg.training.reduce_lr_cooldown
        )
    elif cfg.training.scheduler_type == 'cosine':
        if cfg.training.max_steps is None:
            raise ValueError("max_steps must be specified for cosine scheduler")
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training.max_steps, eta_min=0)
    else:
        raise ValueError(f"Unknown scheduler_type: {cfg.training.scheduler_type}")
    
    return optimizer, scheduler

def train_one_epoch(diffusion, train_dataloader, optimizer, scheduler, cfg, logger, global_step):
    """
    Train the diffusion model for one epoch.

    Args:
        diffusion: The diffusion model.
        train_dataloader: DataLoader for the training dataset.
        optimizer: Optimizer for updating model parameters.
        scheduler: Learning rate scheduler.
        cfg (DictConfig): Hydra configuration object.

    Returns:
        train_mean_loss: Average training loss for the epoch.
    """
    diffusion.train()
    total_loss = 0.0

    with tqdm(train_dataloader, desc="Training", leave=False) as pbar:
        for img, mask, *_ in pbar:
            # Move data to the appropriate device
            img, mask = img.to(cfg.device), mask.to(cfg.device)

            optimizer.zero_grad()
            loss, sample_mses, ts = diffusion.forward(mask, conditioned_image=img)

            loss.backward()
            grad_norm = calc_grad_norm(diffusion.parameters())
            optimizer.step()
            param_norm = calc_param_norm(diffusion.parameters())

            # logging
            batch_size = mask.shape[0]
            global_step += 1
            samples = global_step * batch_size

            logger.logkv_mean('loss', loss.item(), accumulator='train')
            logger.logkv_mean('grad_norm', grad_norm, accumulator='train')
            logger.logkv_mean('param_norm', param_norm, accumulator='train')
            logger.logkv('step', global_step, accumulator='train')
            logger.logkv('samples', samples, accumulator='train')
            logger.logkv('lr', optimizer.param_groups[0]['lr'], accumulator='train')
            logger.logkv_loss_quartiles(diffusion, ts, {'loss': sample_mses}, accumulator='train')

            if global_step % logger.log_interval == 0:
                logger.dumpkvs(global_step, accumulator='train')

            total_loss += loss.item()

            pbar.set_postfix(loss=loss.item())

    # Compute average loss and update scheduler
    train_mean_loss = total_loss / len(train_dataloader)
    scheduler.step(train_mean_loss)

    return train_mean_loss, global_step

@device_grad_decorator(no_grad=True)
def test_one_epoch(diffusion, test_dataloader, cfg, logger, global_step):
    """
    Evaluate the diffusion model for one epoch.

    Args:
        diffusion: The diffusion model.
        test_dataloader: DataLoader for the testing dataset.
        cfg (DictConfig): Hydra configuration object.

    Returns:
        test_mean_loss: Average testing loss for the epoch.
    """
    diffusion.eval()
    total_loss = 0.0

    with tqdm(test_dataloader, desc="Testing", leave=False) as pbar:
        for img, mask, *_ in pbar:
            # Move data to the appropriate device
            img, mask = img.to(cfg.device), mask.to(cfg.device)

            loss, sample_mses, ts = diffusion.forward(mask, conditioned_image=img)
            batch_size = mask.shape[0]
            global_step += 1
            samples = global_step * batch_size

            logger.logkv_mean('test_loss', loss.item(), accumulator='val')
            logger.logkv('test_step', global_step, accumulator='val')
            logger.logkv('test_samples', samples, accumulator='val')
            logger.logkv_loss_quartiles(diffusion, ts, {'test_loss': sample_mses}, accumulator='val')

            if global_step % logger.log_interval == 0:
                logger.dumpkvs(global_step, accumulator='val')

            total_loss += loss.item()

            pbar.set_postfix(loss=loss.item())

    test_mean_loss = total_loss / len(test_dataloader)
    return test_mean_loss, global_step

@device_grad_decorator(no_grad=True)
def validate_one_epoch(diffusion, val_dl, metrics, logger, global_step, cfg):
    diffusion.eval()
    pbar = tqdm(val_dl, desc="Validation", leave=True)
    for img, true_mask, _ in pbar:
        img = img.to(cfg.device)
        true_mask = true_mask.to(cfg.device)
        # Generate predictions (assume diffusion.sample returns predicted masks [B, 1, H, W])
        pred_mask = diffusion.sample(img, disable_tqdm=True)
        
        # Process each sample in batch (since metrics are slice-wise)
        batch_size = img.shape[0]
        for i in range(batch_size):
            pred = pred_mask[i]
            true = true_mask[i]
            for metric in metrics:
                metric(pred, true)  # Call forward
        
        # Compute current running metrics and update postfix
        current_results = {}
        for metric in metrics:
            metric_results = metric.compute()
            if isinstance(metric_results, dict):
                current_results.update(metric_results)
            else:
                current_results[metric.__class__.__name__] = metric_results
        
        key_metrics = {k: f"{current_results.get(k, 0):.4f}" for k in ['dice_2d_fg', 'f1_2d']}
        pbar.set_postfix(**key_metrics)

    # Compute aggregated results
    results = {}
    for metric in metrics:
        metric_results = metric.compute()
        if isinstance(metric_results, dict):  # For aggregators
            results.update(metric_results)
        else:
            results[metric.__class__.__name__] = metric_results

    # Update tqdm postfix with key metrics
    key_metrics = {k: f"{v:.4f}" for k, v in results.items() if k in ['dice_2d_fg', 'f1_2d']}
    pbar.set_postfix(**key_metrics)

    # Reset metrics
    for metric in metrics:
        metric.reset()

    return results

def save_model(diffusion, old_best_epoch, new_best_epoch, model_save_path_template):
    """
    Save the current best model, removing the previous best model file.

    Args:
        diffusion: The diffusion model.
        old_best_epoch: The epoch number of the previous best model.
        new_best_epoch: The epoch number of the current best model.
        model_save_path_template: String template for model save paths.
    """
    # Remove old best model if it exists
    old_model_path = model_save_path_template.format(old_best_epoch)
    if os.path.exists(old_model_path):
        os.remove(old_model_path)

    # Save the new best model
    new_model_path = model_save_path_template.format(new_best_epoch)
    torch.save(diffusion.state_dict(), new_model_path)

def train_and_evaluate(
    cfg,
    diffusion,
    train_dataloader,
    test_dataloader,
    optimizer,
    scheduler,
    logger,
    model_save_path_template=None,
):
    """
    Train and evaluate the diffusion model.

    Args:
        cfg (DictConfig): Hydra configuration object.
        diffusion: The diffusion model.
        train_dataloader: DataLoader for the training dataset.
        test_dataloader: DataLoader for the test dataset.
        optimizer: Optimizer for training.
        scheduler: Learning rate scheduler.
        writer: TensorBoard SummaryWriter (optional).
        model_save_path_template: String template for model save paths.

    Returns:
        train_losses: List of average training losses for each epoch.
        test_losses: List of average test losses for each epoch.
        best_epoch: The epoch with the best test loss.
    """
    best_test_loss = float('inf')
    best_test_loss_epoch = -1
    train_losses = []
    test_losses = []

    global_step = 0

    for epoch in range(1, cfg.training.num_epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.training.num_epochs}")
        print("-" * 30)

        # Train for one epoch
        train_loss, global_step = train_one_epoch(diffusion, train_dataloader, optimizer, scheduler, cfg, logger, global_step)
        train_losses.append(train_loss)

        # Evaluate for one epoch
        test_loss, global_step = test_one_epoch(diffusion, test_dataloader, cfg, logger, global_step)
        test_losses.append(test_loss)

        if logger.writer is not None:
            logger.writer.add_scalar('Loss/train', train_loss, epoch)
            logger.writer.add_scalar('Loss/test', test_loss, epoch)
            logger.writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

        # Save model if it's the best test loss
        if test_loss < best_test_loss:
            save_model(diffusion, best_test_loss_epoch, epoch, model_save_path_template)
            best_test_loss = test_loss
            best_test_loss_epoch = epoch

        # Log progress
        print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
              f"Best Test Loss: {best_test_loss:.4f} (Epoch {best_test_loss_epoch})")

        # Visualization
        if epoch % cfg.training.visualization_sample_interval == 0 or epoch == cfg.training.num_epochs:
            '''
            sample_and_visualize(
                diffusion,
                test_dataloader.dataset,
                num_samples=cfg.training.visualization_num_samples,
                device=cfg.device,
            )
            '''
            pass

    print(f"\nTraining complete. Best Test Loss: {best_test_loss:.4f} at Epoch {best_test_loss_epoch}.")
    if logger.writer is not None:
        logger.writer.close()
    return train_losses, test_losses, best_test_loss_epoch

def step_based_train(cfg, diffusion, dataloaders, optimizer, scheduler, logger, run_dir=None):
    OmegaConf.set_struct(cfg, False)
    # Temporary aliases for config transition
    # Optimizer/scheduler (some may be set in get_optimizer)
    cfg.training.scheduler_interval = cfg.optimizer.scheduler_interval
    cfg.training.scheduler_type = cfg.optimizer.scheduler_type
    
    # Diffusion
    cfg.training.timesteps = cfg.diffusion.timesteps
    
    # Training loop
    cfg.training.max_steps = cfg.training.max_steps
    cfg.training.checkpoint_save_interval = cfg.training.checkpoint_save_interval
    cfg.training.ema_rate = cfg.training.ema_rate
    cfg.training.ema_rate_precision = cfg.training.ema_rate_precision
    
    # Validation/logging
    cfg.validation.validation_interval = cfg.validation.validation_interval
    
    # Dataset (for modalities)
    cfg.dataset.modalities = cfg.dataset.modalities  # Retained or from base
    
    # Environment (device from main, but alias if needed)
    cfg.device = cfg.environment.device

    # Access dataloaders
    train_dataloader = dataloaders['train']
    sample_dataloader = dataloaders['sample']  # Formerly test_dataloader
    val_dataloader = dataloaders['val']  # For future validation

    # Use cfg values directly (remove params)
    max_steps = cfg.training.max_steps
    save_interval = cfg.training.checkpoint_save_interval
    log_interval = cfg.logging.interval
    ema_rates = [float(r) for r in cfg.training.ema_rate.split(',') if r.strip()]
    scheduler_interval = cfg.training.scheduler_interval
    scheduler_type = cfg.training.scheduler_type
    
    diffusion.train()
    global_step = 0
    train_iterator = iter(train_dataloader)
    
    # EMA setup (inspired by MedSegDiff)
    ema_params = [copy.deepcopy(list(diffusion.parameters())) for _ in ema_rates] if ema_rates else []
    
    # Scheduling refinements
    interval_loss = 0.0
    interval_count = 0
    
    while max_steps is None or global_step < max_steps:
        try:
            img, mask, *_ = next(train_iterator)
        except StopIteration:
            # Reinitialize iterator if dataset is exhausted
            train_iterator = iter(train_dataloader)
            img, mask, *_ = next(train_iterator)
        
        img, mask = img.to(cfg.device), mask.to(cfg.device)
        batch_size = mask.shape[0]  # Define here, after data load
        
        optimizer.zero_grad()
        loss, sample_mses, ts = diffusion.forward(mask, conditioned_image=img)
        
        loss.backward()
        grad_norm = calc_grad_norm(diffusion.parameters())
        optimizer.step()
        param_norm = calc_param_norm(diffusion.parameters())
        
        # Update EMAs (after optimizer step)
        for rate, params in zip(ema_rates, ema_params):
            for p, ema_p in zip(diffusion.parameters(), params):
                ema_p.data = (1.0 - rate) * p.data + rate * ema_p.data
        
        # Increment global_step here, after training but before logging
        global_step += 1
        samples = global_step * batch_size
        
        # Validation check
        if global_step % cfg.validation.validation_interval == 0 and global_step > 0:
            # Free memory before validation
            gc.collect()
            torch.cuda.empty_cache()
            
            metrics = [get_metric(m['name'], m.get('params', {})) for m in cfg.validation.metrics]
            val_results = validate_one_epoch(diffusion, val_dataloader, metrics, logger, global_step, cfg)
            logger.log_metrics_dict("val", val_results, global_step, accumulator='val')
            logger.dumpkvs(global_step, accumulator='val')
            logger.clear_accumulators(accumulator='val')
            
            # Free memory after validation
            gc.collect()
            torch.cuda.empty_cache()
        
        # New: Image-based logging
        if cfg.logging.enable_image_logging and global_step % cfg.logging.image_log_interval == 0 and global_step >= cfg.logging.min_log_step:
            print(f"Starting forward image logging at step {global_step}")
            # Log forward process (accumulate fresh batches from dataloader)
            with torch.no_grad():
                collected_masks = []
                collected_imgs = []
                dl_iter = iter(train_dataloader)  # Use iterator
                while len(collected_imgs) < cfg.logging.num_log_samples:
                    try:
                        batch = next(dl_iter)
                    except StopIteration:
                        dl_iter = iter(train_dataloader)  # Restart if exhausted
                        batch = next(dl_iter)
                    b_img, b_mask = batch[0], batch[1]  # Full batch
                    for j in range(b_img.shape[0]):
                        if len(collected_imgs) >= cfg.logging.num_log_samples:
                            break
                        if b_mask[j].sum() > 0:  # Only add non-empty
                            collected_imgs.append(b_img[j:j+1])
                            collected_masks.append(b_mask[j:j+1])
                full_mask = torch.cat(collected_masks, dim=0).to(cfg.device)
                full_img = torch.cat(collected_imgs, dim=0).to(cfg.device)
                num_samples = len(full_mask)
                
                if num_samples < cfg.logging.num_log_samples:
                    print(f"Warning: Accumulated {num_samples} < num_log_samples {cfg.logging.num_log_samples} for forward.")
                
                loss, sample_mses, t, intermediates = diffusion.forward(full_mask, full_img, return_intermediates=True)
                
                all_images = []
                labels = []
                metrics = {}
                num_modalities = len(cfg.dataset.modalities)
                for i in range(num_samples):
                    sample_images = [intermediates['img'][i], intermediates['mask'][i], intermediates['x_t'][i], intermediates['noise'][i], intermediates['noise_hat'][i]]
                    all_images.extend(sample_images)
                    
                    modality_labels = [f"Modality: {cfg.dataset.modalities[j]}" for j in range(num_modalities)]
                    noisy_label = f"Noise Mask t={t[i].item()}"
                    base_labels = modality_labels + ["Target Mask", noisy_label, "True Noise", "Predicted Noise"]
                    labels.extend(base_labels)
                    
                    metrics[i] = sample_mses[i].item()
                
                logger.log_image_grid(f'Training/Forward_Step_{global_step}', all_images, global_step, metrics, 'horizontal', labels=labels, per_sample_ncol=len(base_labels))
            
        # Sampling logging (outside image logging block)
        if cfg.logging.enable_sampling_snapshots and global_step % cfg.logging.sampling_log_interval == 0 and global_step >= cfg.logging.min_log_step:
            print(f"Starting sampling logging at step {global_step}")
            
            # Select dataloader
            if sample_dataloader is None and cfg.logging.log_test_samples:
                print("Warning: sample_dataloader is None but log_test_samples=true; falling back to train_dataloader.")
            dl = sample_dataloader if cfg.logging.log_test_samples and sample_dataloader is not None else train_dataloader
            if dl is None:
                print("Warning: No dataloader available for sampling – skipping snapshot logging.")
            else:
                # Accumulate samples (images + masks) until we have num_log_samples
                collected_imgs, collected_masks = [], []
                remaining = cfg.logging.num_log_samples
                dl_iter = iter(dl)  # Use iterator to avoid recreating loader each time
                while len(collected_imgs) < cfg.logging.num_log_samples:
                    try:
                        b_img, b_mask, *_ = next(dl_iter)
                    except StopIteration:
                        dl_iter = iter(dl)  # Restart if exhausted
                        b_img, b_mask, *_ = next(dl_iter)
                    for j in range(b_img.shape[0]):
                        if len(collected_imgs) >= cfg.logging.num_log_samples:
                            break
                        if b_mask[j].sum() > 0:  # Only add non-empty
                            collected_imgs.append(b_img[j:j+1])
                            collected_masks.append(b_mask[j:j+1])
                sample_imgs = torch.cat(collected_imgs, dim=0).to(cfg.device)
                sample_masks = torch.cat(collected_masks, dim=0).to(cfg.device)
                num_samples = sample_imgs.shape[0]
                
                # Build composite grid
                all_images = []
                labels = []
                num_modalities = sample_imgs.shape[1]
                # Determine timesteps that sample_with_snapshots will yield
                snapshots_per_sample = []
                for i in range(num_samples):
                    snaps = list(diffusion.sample_with_snapshots(sample_imgs[i:i+1], cfg.logging.snapshot_step_interval))
                    # snaps is list of (t, mask); keep order as yielded (descending t) then final 0
                    snapshots_per_sample.append([normalize_to_neg_one_to_one(m[1]) for m in snaps])  # Normalize snapshots to -1/1
                num_snapshots = len(snapshots_per_sample[0])
                
                # Labels per category
                modality_labels = [f"Modality: {cfg.dataset.modalities[j]}" for j in range(num_modalities)]
                snapshot_labels = [f"t={cfg.training.timesteps - s*cfg.logging.snapshot_step_interval}" for s in range(num_snapshots-1)] + ["t=0"]
                row_len = num_modalities + num_snapshots + 1  # +1 for target
                
                for i in range(num_samples):
                    # append each modality channel individually
                    for c in range(num_modalities):
                        mod_channel = normalize_to_neg_one_to_one(sample_imgs[i, c:c+1])  # Normalize to -1/1 for black empty
                        all_images.append(mod_channel)
                        labels.append(modality_labels[c])
                    # snapshots
                    for snap in snapshots_per_sample[i]:
                        all_images.append(snap)
                    labels.extend(snapshot_labels)
                    # target mask (normalize to match forward)
                    target_norm = normalize_to_neg_one_to_one(sample_masks[i])
                    all_images.append(target_norm)
                    labels.append("Target")
                
                logger.log_image_grid(
                    f"Sampling/Snapshots_Step_{global_step}",
                    all_images,
                    global_step,
                    metrics=None,
                    grid_layout='horizontal',
                    labels=labels,
                    per_sample_ncol=row_len,
                )
        
        # Logging (scalars)
        logger.logkv_mean('loss', loss.item(), accumulator='train')
        logger.logkv_mean('grad_norm', grad_norm, accumulator='train')
        logger.logkv_mean('param_norm', param_norm, accumulator='train')
        logger.logkv('step', global_step, accumulator='train')
        logger.logkv('samples', samples, accumulator='train')
        logger.logkv('lr', optimizer.param_groups[0]['lr'], accumulator='train')
        logger.logkv_loss_quartiles(diffusion, ts, {'loss': sample_mses}, accumulator='train')
        
        if global_step % log_interval == 0:
            logger.dumpkvs(global_step, accumulator='train')
            logger.clear_accumulators(accumulator='train')
        
        # Accumulate for scheduling
        interval_loss += loss.item()
        interval_count += 1
        
        if global_step % scheduler_interval == 0:
            if interval_count > 0:
                mean_loss = interval_loss / interval_count
                prev_lr = optimizer.param_groups[0]['lr']
                prev_best = scheduler.best
                prev_bad_epochs = scheduler.num_bad_epochs
                prev_cooldown = scheduler.cooldown_counter
                
                scheduler.step(mean_loss) if scheduler_type == 'reduce_lr' else scheduler.step()
                
                new_lr = optimizer.param_groups[0]['lr']
                reduced = new_lr < prev_lr
                
                print(f"Scheduler Update at step {global_step}:")
                print(f"  Configured Threshold: {scheduler.threshold:.6e}")
                print(f"  Configured Cooldown: {scheduler.cooldown}")
                print(f"  Mean Loss: {mean_loss:.6f}")
                print(f"  Previous Best: {prev_best:.6f}")
                print(f"  New Best: {scheduler.best:.6f}")
                print(f"  Bad Epochs: {prev_bad_epochs} -> {scheduler.num_bad_epochs}")
                print(f"  Cooldown: {prev_cooldown} -> {scheduler.cooldown_counter}")
                print(f"  LR: {prev_lr:.6e} -> {new_lr:.6e} (Reduced: {reduced})")
                
            interval_loss = 0.0
            interval_count = 0
        
        # Periodic saving
        if global_step % save_interval == 0 and global_step > 0:
            save_checkpoint(diffusion, optimizer, ema_params, ema_rates, global_step, cfg, run_dir)
        
        # Removed old visualization block
        
    # Final save and log
    save_checkpoint(diffusion, optimizer, ema_params, ema_rates, global_step, cfg, run_dir)
    logger.dumpkvs(global_step, accumulator='train')
    logger.clear_accumulators(accumulator='train')
    
    print(f"Step-based training complete at step {global_step}.")

def save_checkpoint(diffusion, optimizer, ema_params, ema_rates, step, cfg, run_dir=None):  # Pass cfg for templates
    OmegaConf.set_struct(cfg, False)
    # Aliases for templates
    cfg.training.main_checkpoint_template = cfg.training.main_checkpoint_template
    cfg.training.ema_checkpoint_template = cfg.training.ema_checkpoint_template
    cfg.training.opt_checkpoint_template = cfg.training.opt_checkpoint_template
    cfg.training.ema_rate_precision = cfg.training.ema_rate_precision

    # Save main model
    main_path = f"{run_dir}{cfg.training.main_checkpoint_template.format(step)}"
    os.makedirs(os.path.dirname(main_path), exist_ok=True)
    torch.save(diffusion.state_dict(), main_path)
    print(f"Saved model checkpoint to {main_path} at step {step}")
    
    # Save EMAs (one per rate)
    for rate, params in zip(ema_rates, ema_params):
        formatted_rate = f"{rate:.{cfg.training.ema_rate_precision}f}"
        ema_path = f"{run_dir}{cfg.training.ema_checkpoint_template.format(step, rate=formatted_rate)}"
        ema_state = {k: v for k, v in zip(diffusion.state_dict().keys(), (p.data for p in params))}
        torch.save(ema_state, ema_path)
        print(f"Saved EMA (rate {rate}) checkpoint to {ema_path} at step {step}")
    
    # Save optimizer
    opt_path = f"{run_dir}{cfg.training.opt_checkpoint_template.format(step)}"
    torch.save(optimizer.state_dict(), opt_path)
    print(f"Saved optimizer checkpoint to {opt_path} at step {step}")
    OmegaConf.set_struct(cfg, True)
