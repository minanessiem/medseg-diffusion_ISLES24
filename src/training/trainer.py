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
from src.models.architectures.unet_util import unnormalize_to_zero_to_one

def get_optimizer_and_scheduler(cfg, model):
    """
    Create optimizer and learning rate scheduler based on configuration.

    Args:
        cfg (DictConfig): Hydra configuration object.
        model (nn.Module): The model to optimize.

    Returns:
        optimizer, scheduler
    """
    optimizer = Adam(model.parameters(), lr=cfg.training.learning_rate)
    
    if cfg.training.scheduler_type == 'reduce_lr':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=cfg.training.reduce_lr_factor,
            patience=cfg.training.reduce_lr_patience
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
            loss, sample_mses, ts = diffusion(mask, img)

            loss.backward()
            grad_norm = calc_grad_norm(diffusion.parameters())
            optimizer.step()
            param_norm = calc_param_norm(diffusion.parameters())

            # logging
            batch_size = mask.shape[0]
            global_step += 1
            samples = global_step * batch_size

            logger.logkv_mean('loss', loss.item())
            logger.logkv_mean('grad_norm', grad_norm)
            logger.logkv_mean('param_norm', param_norm)
            logger.logkv('step', global_step)
            logger.logkv('samples', samples)
            logger.logkv('lr', optimizer.param_groups[0]['lr'])
            logger.logkv_loss_quartiles(diffusion, ts, {'loss': sample_mses})

            if global_step % logger.log_interval == 0:
                logger.dumpkvs(global_step)

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

            loss, sample_mses, ts = diffusion(mask, img)
            batch_size = mask.shape[0]
            global_step += 1
            samples = global_step * batch_size

            logger.logkv_mean('test_loss', loss.item())
            logger.logkv('test_step', global_step)
            logger.logkv('test_samples', samples)
            logger.logkv_loss_quartiles(diffusion, ts, {'test_loss': sample_mses})

            if global_step % logger.log_interval == 0:
                logger.dumpkvs(global_step)

            total_loss += loss.item()

            pbar.set_postfix(loss=loss.item())

    test_mean_loss = total_loss / len(test_dataloader)
    return test_mean_loss, global_step

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

def step_based_train(
    cfg,
    diffusion,
    train_dataloader,
    optimizer,
    scheduler,
    logger,
    model_save_path_template=None,
    max_steps: int = None,
    save_interval: int = 1000,
    log_interval: int = 100,
    test_dataloader = None,  # New optional param for test sampling
):
    # Use config values, assuming they exist
    save_interval = cfg.training.checkpoint_save_interval
    log_interval = cfg.logging.interval  # Assuming from logging config
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
        
        optimizer.zero_grad()
        loss, sample_mses, ts = diffusion(mask, img)
        
        loss.backward()
        grad_norm = calc_grad_norm(diffusion.parameters())
        optimizer.step()
        param_norm = calc_param_norm(diffusion.parameters())
        
        # Update EMAs (after optimizer step)
        for rate, params in zip(ema_rates, ema_params):
            for p, ema_p in zip(diffusion.parameters(), params):
                ema_p.data = (1.0 - rate) * p.data + rate * ema_p.data
        
        # New: Image-based logging
        if cfg.logging.enable_image_logging and global_step % cfg.logging.image_log_interval == 0:
            # Log forward process (reuse current batch for efficiency)
            with torch.no_grad():
                num_available = len(mask)
                if num_available < cfg.logging.num_log_samples:
                    print(f"Warning: Batch size {num_available} < num_log_samples {cfg.logging.num_log_samples}; logging {num_available} for forward.")
                num_samples = min(cfg.logging.num_log_samples, num_available)
                loss, sample_mses, t, intermediates = diffusion(mask[:num_samples], img[:num_samples], return_intermediates=True)
                for i in range(num_samples):
                    sample_images = [intermediates['img'][i], intermediates['mask'][i], intermediates['x_t'][i], intermediates['noise'][i], intermediates['noise_hat'][i]]
                    metrics = {0: sample_mses[i].item()}
                    logger.log_image_grid(f'Training/Forward_Step_{global_step}_sample{i}', sample_images, global_step, metrics, 'horizontal')
        
        if cfg.logging.enable_sampling_snapshots and global_step % cfg.logging.sampling_log_interval == 0:
            # Sample from test or train, accumulating multiple batches if needed
            dl = test_dataloader if cfg.logging.log_test_samples and test_dataloader else train_dataloader
            collected_imgs = []
            remaining = cfg.logging.num_log_samples
            while remaining > 0:
                batch = next(iter(dl))
                batch_imgs = batch[0][:remaining]  # Take up to remaining
                collected_imgs.append(batch_imgs)
                remaining -= len(batch_imgs)
            
            sample_img = torch.cat(collected_imgs, dim=0).to(cfg.device)
            num_samples = len(sample_img)
            
            for i in range(num_samples):
                snapshots = []
                for t, mask_snapshot in diffusion.sample_with_snapshots(sample_img[i:i+1], cfg.logging.snapshot_step_interval):
                    snapshots.append((t, mask_snapshot[0]))
                
                for j, (t, mask_snap) in enumerate(snapshots):
                    grid_images = [sample_img[i], mask_snap]
                    logger.log_image_grid(f'Sampling/Snapshots_Step_{global_step}_sample{i}_t{t}', grid_images, global_step, grid_layout='vertical')
        
        # Logging
        batch_size = mask.shape[0]
        global_step += 1
        samples = global_step * batch_size
        
        logger.logkv_mean('loss', loss.item())
        logger.logkv_mean('grad_norm', grad_norm)
        logger.logkv_mean('param_norm', param_norm)
        logger.logkv('step', global_step)
        logger.logkv('samples', samples)
        logger.logkv('lr', optimizer.param_groups[0]['lr'])
        logger.logkv_loss_quartiles(diffusion, ts, {'loss': sample_mses})
        
        if global_step % log_interval == 0:
            logger.dumpkvs(global_step)
            logger.clear_accumulators()  # Ensure reset after dump
        
        # Accumulate for scheduling
        interval_loss += loss.item()
        interval_count += 1
        
        if global_step % scheduler_interval == 0:
            if interval_count > 0:
                mean_loss = interval_loss / interval_count
                scheduler.step(mean_loss) if scheduler_type == 'reduce_lr' else scheduler.step()
            interval_loss = 0.0
            interval_count = 0
        
        # Periodic saving
        if global_step % save_interval == 0 and global_step > 0:
            save_checkpoint(diffusion, optimizer, ema_params, ema_rates, global_step, cfg)
        
        # Removed old visualization block
        
    # Final save and log
    save_checkpoint(diffusion, optimizer, ema_params, ema_rates, global_step, cfg)
    logger.dumpkvs(global_step)
    logger.clear_accumulators()
    
    print(f"Step-based training complete at step {global_step}.")

def save_checkpoint(diffusion, optimizer, ema_params, ema_rates, step, cfg):  # Pass cfg for templates
    # Save main model
    main_path = cfg.training.main_checkpoint_template.format(step)
    os.makedirs(os.path.dirname(main_path), exist_ok=True)
    torch.save(diffusion.state_dict(), main_path)
    print(f"Saved model checkpoint to {main_path} at step {step}")
    
    # Save EMAs (one per rate)
    for rate, params in zip(ema_rates, ema_params):
        formatted_rate = f"{rate:.{cfg.training.ema_rate_precision}f}"
        ema_path = cfg.training.ema_checkpoint_template.format(step, rate=formatted_rate)
        ema_state = {k: v for k, v in zip(diffusion.state_dict().keys(), (p.data for p in params))}
        torch.save(ema_state, ema_path)
        print(f"Saved EMA (rate {rate}) checkpoint to {ema_path} at step {step}")
    
    # Save optimizer
    opt_path = cfg.training.opt_checkpoint_template.format(step)
    torch.save(optimizer.state_dict(), opt_path)
    print(f"Saved optimizer checkpoint to {opt_path} at step {step}")
