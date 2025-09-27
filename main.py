import hydra
from omegaconf import DictConfig
import torch
import random
import numpy as np

from src.models.architectures.unet import Unet
from src.models.components.diffusion import Diffusion
from src.data.loaders import BrainMRIDataset, get_dataloaders  # Assuming get_dataloaders is implemented
from src.training.trainer import get_optimizer_and_scheduler, train_and_evaluate
from src.evaluation.evaluator import plot_losses, load_best_model, visualize_best_model_predictions, visualize_noise_schedulers  # Optional
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig):
    # Set seeds for reproducibility (from notebook Cell 4)
    torch.manual_seed(cfg.random_seed)
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    # Device configuration (auto-detect with cfg override)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if cfg.device == 'auto' else torch.device(cfg.device)
    print(f'Using device: {device}')

    # Optional: Visualize noise schedulers
    # visualize_noise_schedulers(cfg)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"unet_img{cfg.model.image_size}_lr{cfg.training.learning_rate}_epochs{cfg.training.num_epochs}_steps{cfg.training.timesteps}_{timestamp}"
    writer = None
    model_save_path_template = cfg.training.model_save_path_template  # Default from config
    if cfg.mode == "train":
        run_output_dir = f"{cfg.training.output_root}{run_name}/"
        os.makedirs(f"{run_output_dir}tensorboard/", exist_ok=True)
        os.makedirs(f"{run_output_dir}{cfg.training.model_save_dir}", exist_ok=True)  # Use config's relative model_save_dir
        writer = SummaryWriter(log_dir=f"{run_output_dir}tensorboard/")
        model_save_path_template = f"{run_output_dir}{cfg.training.model_save_path_template}"  # Prepend run dir to template

    # Build model
    unet = Unet(cfg).to(device)
    diffusion = Diffusion(model=unet, cfg=cfg, device=device).to(device)

    # Get dataloaders
    train_dl, test_dl = get_dataloaders(cfg)

    if cfg.mode == "train":
        optimizer, scheduler = get_optimizer_and_scheduler(cfg, diffusion)
        train_losses, test_losses, best_epoch = train_and_evaluate(
            cfg,
            diffusion,
            train_dl,
            test_dl,
            optimizer,
            scheduler,
            writer=writer,
            model_save_path_template=model_save_path_template,  # New param for dynamic path
        )
        plot_losses(train_losses, test_losses)
    elif cfg.mode == "evaluate":
        # Assuming best_epoch is known or passed via cfg; for demo, hard-code or load from file
        best_epoch = cfg.get('best_epoch', 50)  # Placeholder; in practice, track this
        diffusion = load_best_model(diffusion, cfg, best_epoch)
        visualize_best_model_predictions(diffusion, test_dl.dataset, cfg)

if __name__ == '__main__':
    main()