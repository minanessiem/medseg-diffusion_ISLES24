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
from src.utils.logger import Logger
import os
from datetime import datetime

# Utility to parse multi_gpu flag

def _parse_multi_gpu_flag(flag):
    """Return list of GPU ids or None."""
    if not flag or str(flag).lower() in {"false", "none", ""}:
        return None
    if isinstance(flag, (list, tuple)):
        return [int(x) for x in flag]
    return [int(x.strip()) for x in str(flag).split(',') if x.strip()]

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
    log_dir = "runs/"  # default
    if cfg.mode == "train":
        run_output_dir = f"{cfg.training.output_root}{run_name}/"
        os.makedirs(f"{run_output_dir}tensorboard/", exist_ok=True)
        os.makedirs(f"{run_output_dir}{cfg.training.model_save_dir}", exist_ok=True)
        writer = SummaryWriter(log_dir=f"{run_output_dir}tensorboard/")
        log_dir = f"{run_output_dir}tensorboard/"
        model_save_path_template = f"{run_output_dir}{cfg.training.model_save_path_template}"
    else:
        # In evaluate mode, still create a writer for consistency (logs under runs/eval-<timestamp>)
        log_dir = f"runs/eval_{timestamp}/"
        writer = SummaryWriter(log_dir=log_dir)

    # Initialize Logger with hydra config
    logger = Logger(
        log_dir=log_dir,
        enabled_outputs=list(cfg.logging.outputs),
        log_interval=int(cfg.logging.interval),
        table_format=cfg.logging.table_format,
        writer=writer,
    )

    # Build model
    unet = Unet(cfg).to(device)
    diffusion = Diffusion(model=unet, cfg=cfg, device=device)

    # Multi-GPU handling via DataParallel
    gpu_ids = _parse_multi_gpu_flag(cfg.training.multi_gpu)
    if gpu_ids:
        visible = torch.cuda.device_count()
        if max(gpu_ids) >= visible:
            raise ValueError(f"Requested GPU id {max(gpu_ids)} but only {visible} GPUs visible.")
        print(f"Using GPUs {gpu_ids} with torch.nn.DataParallel")
        diffusion = torch.nn.DataParallel(diffusion, device_ids=gpu_ids).cuda(gpu_ids[0])
    else:
        diffusion = diffusion.to(device)

    # Get dataloaders
    train_dl, test_dl = get_dataloaders(cfg)

    if cfg.dataset.name == 'isles24':
        assert cfg.model.image_channels == len(cfg.dataset.modalities), "Model image_channels must match number of modalities"

    if cfg.mode == "train":
        optimizer, scheduler = get_optimizer_and_scheduler(cfg, diffusion)
        train_losses, test_losses, best_epoch = train_and_evaluate(
            cfg,
            diffusion,
            train_dl,
            test_dl,
            optimizer,
            scheduler,
            logger=logger,
            model_save_path_template=model_save_path_template,
        )
        plot_losses(train_losses, test_losses)
    elif cfg.mode == "evaluate":
        # Assuming best_epoch is known or passed via cfg; for demo, hard-code or load from file
        best_epoch = cfg.get('best_epoch', 50)  # Placeholder; in practice, track this
        diffusion = load_best_model(diffusion, cfg, best_epoch)
        visualize_best_model_predictions(diffusion, test_dl.dataset, cfg)
    logger.close()

if __name__ == '__main__':
    main()