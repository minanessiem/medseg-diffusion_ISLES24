import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.utils.general import device_grad_decorator
from src.evaluation.evaluator import sample_and_visualize  # For visualization in train_and_evaluate
from src.utils.logger import Logger
from src.utils.train_utils import calc_grad_norm, calc_param_norm

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
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=cfg.training.reduce_lr_factor,
        patience=cfg.training.reduce_lr_patience
    )
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
            sample_and_visualize(
                diffusion,
                test_dataloader.dataset,
                num_samples=cfg.training.visualization_num_samples,
                device=cfg.device,
            )

    print(f"\nTraining complete. Best Test Loss: {best_test_loss:.4f} at Epoch {best_test_loss_epoch}.")
    if logger.writer is not None:
        logger.writer.close()
    return train_losses, test_losses, best_test_loss_epoch
