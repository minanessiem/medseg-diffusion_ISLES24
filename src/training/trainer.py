import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.utils.general import device_grad_decorator
from src.evaluation.evaluator import sample_and_visualize  # For visualization in train_and_evaluate

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
        patience=cfg.training.reduce_lr_patience,
        verbose=True
    )
    return optimizer, scheduler

def train_one_epoch(diffusion, train_dataloader, optimizer, scheduler, cfg):
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

    for img, mask, label in tqdm(train_dataloader, desc="Training", leave=False):
        # Move data to the appropriate device
        img, mask = img.to(cfg.device), mask.to(cfg.device)

        # Forward pass and compute loss
        optimizer.zero_grad()
        loss = diffusion(mask, img)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()

    # Compute average loss and update scheduler
    train_mean_loss = total_loss / len(train_dataloader)
    scheduler.step(train_mean_loss)

    return train_mean_loss

@device_grad_decorator(no_grad=True)
def test_one_epoch(diffusion, test_dataloader, cfg):
    """
    Evaluate the diffusion model for one epoch.

    Args:
        diffusion: The diffusion model.
        test_dataloader: DataLoader for the test dataset.
        cfg (DictConfig): Hydra configuration object.

    Returns:
        test_mean_loss: Average test loss for the epoch.
    """
    diffusion.eval()
    total_loss = 0.0

    for img, mask, label in tqdm(test_dataloader, desc="Testing", leave=False):
        # Move data to the appropriate device
        img, mask = img.to(cfg.device), mask.to(cfg.device)

        # Forward pass and compute loss
        loss = diffusion(mask, img)

        # Accumulate loss
        total_loss += loss.item()

    # Compute average loss
    test_mean_loss = total_loss / len(test_dataloader)

    return test_mean_loss

def save_model(diffusion, old_best_epoch, new_best_epoch, cfg):
    """
    Save the current best model, removing the previous best model file.

    Args:
        diffusion: The diffusion model.
        old_best_epoch: The epoch number of the previous best model.
        new_best_epoch: The epoch number of the current best model.
        cfg (DictConfig): Hydra configuration object with path template.
    """
    path_template = cfg.training.model_save_path_template
    # Remove old best model if it exists
    old_model_path = path_template.format(old_best_epoch)
    if os.path.exists(old_model_path):
        os.remove(old_model_path)

    # Save the new best model
    new_model_path = path_template.format(new_best_epoch)
    torch.save(diffusion.state_dict(), new_model_path)

def train_and_evaluate(
    cfg,
    diffusion,
    train_dataloader,
    test_dataloader,
    optimizer,
    scheduler,
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

    Returns:
        train_losses: List of average training losses for each epoch.
        test_losses: List of average test losses for each epoch.
        best_epoch: The epoch with the best test loss.
    """
    best_test_loss = float('inf')
    best_test_loss_epoch = -1
    train_losses = []
    test_losses = []

    for epoch in range(1, cfg.training.num_epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.training.num_epochs}")
        print("-" * 30)

        # Train for one epoch
        train_loss = train_one_epoch(diffusion, train_dataloader, optimizer, scheduler, cfg)
        train_losses.append(train_loss)

        # Evaluate for one epoch
        test_loss = test_one_epoch(diffusion, test_dataloader, cfg)
        test_losses.append(test_loss)

        # Save model if it's the best test loss
        if test_loss < best_test_loss:
            save_model(diffusion, best_test_loss_epoch, epoch, cfg)
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
    return train_losses, test_losses, best_test_loss_epoch