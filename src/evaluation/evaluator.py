import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils.general import device_grad_decorator, visualize_img
from src.models.components.diffusion import unnormalize_to_zero_to_one  # Assuming it's moved or imported here; adjust if in unet_util
from src.models.components.diffusion import NoiseScheduler  # For visualize_noise_schedulers
import pandas as pd  # For dataframe in noise scheduler vis

def plot_prediction(axs, index, dataset, prediction):
    """
    Plot an MRI image, its corresponding mask, and the predicted mask.

    Args:
        axs: List of matplotlib Axes objects for plotting.
        index: Index of the sample in the dataset.
        dataset: The dataset containing the images and masks.
        prediction: The predicted mask (Tensor).
    """
    # Retrieve the unprocessed image and corresponding label
    img = dataset.get_unprocessed_image(index)
    _, mask, label = dataset[index]

    # Set up the MRI image
    label = int(label)
    status = 'Cancerous' if label == 1 else 'Non-cancerous'
    axs[0].imshow(img)
    axs[0].set_title(f'A {status} MRI')
    axs[0].axis('off')

    # Plot the ground truth mask
    mask = mask.permute(1, 2, 0).numpy()
    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title("Mask")
    axs[1].axis('off')

    # Plot the predicted mask
    prediction = prediction.detach().cpu().permute(1, 2, 0).numpy()
    axs[2].imshow(prediction, cmap='gray')
    axs[2].set_title("Mask Prediction")
    axs[2].axis('off')

@device_grad_decorator(no_grad=True)
def sample_and_visualize(diffusion, dataset, num_samples, cfg):
    """
    Sample and visualize predictions for a subset of the dataset.

    Args:
        diffusion: The diffusion model for generating samples.
        dataset: The dataset containing images and masks.
        num_samples: Number of samples to visualize.
        cfg (DictConfig): Hydra configuration object.
    """
    # Randomly select indices from the dataset
    sampled_indices = np.random.choice(len(dataset), size=num_samples, replace=False)
    
    # Load the sampled images and move them to the specified device
    sampled_images = torch.stack([dataset[i][0] for i in sampled_indices]).to(cfg.device)
    
    # Generate predictions using the diffusion model
    predictions = diffusion.sample(sampled_images)
    
    # Create a figure with subplots
    fig, axs = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    # Ensure axs is always iterable (even for one sample)
    if num_samples == 1:
        axs = [axs]
    
    # Plot each sample's MRI image, mask, and prediction
    for i, sample_idx in enumerate(sampled_indices):
        plot_prediction(axs[i], sample_idx, dataset, predictions[i])
    
    # Adjust layout and show the visualization
    fig.tight_layout()
    plt.show()

def plot_losses(train_losses, test_losses):
    """
    Plot the training and test losses over epochs.

    Args:
        train_losses: List of training losses for each epoch.
        test_losses: List of test losses for each epoch.
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)  # Epochs start at 1 for better clarity

    plt.plot(epochs, train_losses, label='Train Loss', linewidth=2)
    plt.plot(epochs, test_losses, label='Test Loss', linewidth=2)

    plt.title('Training vs Test Loss', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (Log Scale)', fontsize=12)
    plt.yscale('log')  # Logarithmic scale for better visualization
    plt.legend(fontsize=12)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.tight_layout()

    plt.show()

def load_best_model(diffusion, cfg, best_epoch):
    """
    Load the best saved diffusion model from the specified epoch.

    Args:
        diffusion: The diffusion model instance.
        cfg (DictConfig): Hydra configuration object.
        best_epoch: The epoch number of the best model.

    Returns:
        diffusion: The diffusion model with loaded weights.
    """
    best_model_path = cfg.training.model_save_path_template.format(str(int(best_epoch)))
    diffusion.load_state_dict(
        torch.load(
            best_model_path,
            map_location=cfg.device,
        ),
        strict=False,
    )
    diffusion.eval()
    print(f"Best model from epoch {best_epoch} loaded successfully.")
    return diffusion

def visualize_best_model_predictions(diffusion, dataset, cfg):
    """
    Visualize predictions of the best diffusion model on a subset of the dataset.

    Args:
        diffusion: The best diffusion model with loaded weights.
        dataset: The dataset containing images and masks.
        cfg (DictConfig): Hydra configuration object.
    """
    print("\nVisualizing predictions from the best model...")
    sample_and_visualize(diffusion, dataset, cfg.training.visualization_num_samples, cfg)

# Optional: Noise scheduler visualization from Cell 56
def create_noise_scheduler_dataframe(timesteps, beta_min, beta_max, modes):
    """
    Args:
        timesteps (int): Number of timesteps for the schedule.
        beta_min (float): Minimum beta value.
        beta_max (float): Maximum beta value.
        modes (list): List of scheduling modes to generate schedules for.

    Returns:
        pd.DataFrame: DataFrame containing beta, alpha, and alpha_bar schedules.
    """
    df = pd.DataFrame()
    df['timestep'] = np.arange(timesteps)

    for mode in modes:
        scheduler = NoiseScheduler(timesteps, beta_min, beta_max, mode)
        df[f'{mode}_beta'] = scheduler.get_beta_schedule()
        df[f'{mode}_alpha'] = scheduler.get_alpha_schedule()
        df[f'{mode}_alpha_bar'] = scheduler.get_alpha_bar_schedule()

    return df

def plot_scheduler_data(df, x_col, y_cols, titles):
    """
   Args:
        df (pd.DataFrame): DataFrame containing the scheduler data.
        x_col (str): Column name for the x-axis.
        y_cols (list of list): Lists of column names for the y-axis for each subplot.
        titles (list): Titles for each subplot.
    """
    plt.figure(figsize=(18, 6))

    for i, (y_col, title) in enumerate(zip(y_cols, titles), 1):
        plt.subplot(1, len(y_cols), i)
        for col in y_col:
            plt.plot(df[x_col], df[col], label=col)
        plt.xlabel(x_col)
        plt.ylabel('Value')
        plt.title(title)
        plt.legend()

def visualize_noise_schedulers(cfg):
    modes = ['linear', 'cosine', 'quad']  # From notebook
    noise_df = create_noise_scheduler_dataframe(
        timesteps=cfg.training.timesteps,  # Use cfg
        beta_min=0.0001,  # Hard-coded as in notebook; could add to cfg
        beta_max=0.02,
        modes=modes
    )

    y_cols = [
        [f'{mode}_beta' for mode in modes],
        [f'{mode}_alpha' for mode in modes],
        [f'{mode}_alpha_bar' for mode in modes]
    ]
    titles = ['Beta vs Timestep', 'Alpha vs Timestep', 'Alpha Bar vs Timestep']

    plot_scheduler_data(noise_df, 'timestep', y_cols, titles)
    plt.show()