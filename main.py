import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import random
import numpy as np
import shutil

from src.models import build_model
from src.diffusion.diffusion import Diffusion
from src.data.loaders import get_dataloaders
from src.training.trainer import get_optimizer_and_scheduler, step_based_train
from src.evaluation.evaluator import plot_losses, load_best_model, visualize_best_model_predictions, visualize_noise_schedulers  # Optional
from torch.utils.tensorboard import SummaryWriter
from src.utils.logger import Logger
import os
from datetime import datetime
from src.utils.run_name import generate_run_name

# Utility to parse multi_gpu flag

def _parse_multi_gpu_flag(flag):
    """Return list of GPU ids or None."""
    if not flag or str(flag).lower() in {"false", "none", ""}:
        return None
    if isinstance(flag, (list, tuple)):
        return [int(x) for x in flag]
    return [int(x.strip()) for x in str(flag).split(',') if x.strip()]

@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg: DictConfig):
    # Temporary aliases for config transition - remove after full refactor
    OmegaConf.set_struct(cfg, False)  # Temporarily allow dynamic keys
    # Optimizer aliases
    cfg.training.learning_rate = cfg.optimizer.learning_rate
    cfg.training.reduce_lr_factor = cfg.optimizer.reduce_lr_factor
    cfg.training.reduce_lr_patience = cfg.optimizer.reduce_lr_patience
    cfg.training.reduce_lr_threshold = cfg.optimizer.reduce_lr_threshold
    cfg.training.reduce_lr_cooldown = cfg.optimizer.reduce_lr_cooldown
    cfg.training.scheduler_type = cfg.optimizer.scheduler_type
    cfg.training.scheduler_interval = cfg.optimizer.scheduler_interval
    
    # Diffusion aliases
    cfg.training.timesteps = cfg.diffusion.timesteps
    cfg.training.noise_schedule = cfg.diffusion.noise_schedule
    
    # Environment aliases (training)
    cfg.training.output_root = cfg.environment.training.output_root
    cfg.training.model_save_dir = cfg.environment.training.model_save_dir
    cfg.training.multi_gpu = cfg.environment.training.multi_gpu
    
    # Environment aliases (top-level and dataset)
    cfg.device = cfg.environment.device
    cfg.dataset.dir = cfg.environment.dataset.dir
    cfg.dataset.json_list = cfg.environment.dataset.json_list
    cfg.dataset.num_train_workers = cfg.environment.dataset.num_train_workers
    cfg.dataset.num_valid_workers = cfg.environment.dataset.num_valid_workers
    cfg.dataset.num_test_workers = cfg.environment.dataset.num_test_workers
    cfg.dataset.train_batch_size = cfg.environment.dataset.train_batch_size
    cfg.dataset.test_batch_size = cfg.environment.dataset.test_batch_size
    OmegaConf.set_struct(cfg, True)  # Restore struct mode

    # Set seeds for reproducibility (from notebook Cell 4)
    torch.manual_seed(cfg.random_seed)
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    # Device configuration (auto-detect with cfg override)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if cfg.device == 'auto' else torch.device(cfg.device)
    print(f'Using device: {device}')

    # Optional: Visualize noise schedulers
    # visualize_noise_schedulers(cfg)

    # Use overridden timestamp/run_name if provided
    timestamp = cfg.get("timestamp", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    run_name = cfg.get("run_name", generate_run_name(cfg, timestamp))
    
    # Compute and create run_dir
    run_output_dir = f"{cfg.environment.training.output_root}{run_name}/"
    os.makedirs(run_output_dir, exist_ok=True)
    os.makedirs(f"{run_output_dir}tensorboard/", exist_ok=True)
    os.makedirs(f"{run_output_dir}{cfg.training.model_save_dir}", exist_ok=True)

    OmegaConf.set_struct(cfg, False)  # Allow update to hydra key
    # Override Hydra run.dir to run_dir for logs
    OmegaConf.update(cfg, "hydra.run.dir", run_output_dir)
    OmegaConf.set_struct(cfg, True)  # Restore

    # Move early Hydra logs (e.g., main.log) from temp to run_dir
    from hydra.core.hydra_config import HydraConfig
    temp_log_dir = HydraConfig.get().run.dir  # Direct runtime access
    early_log = f"{temp_log_dir}/main.log"
    if os.path.exists(early_log):
        os.makedirs(run_output_dir, exist_ok=True)
        shutil.move(early_log, f"{run_output_dir}/main.log")
        print(f"Moved main.log to {run_output_dir}")

    # Move .hydra/ metadata folder to run_dir
    hydra_dir = f"{temp_log_dir}/.hydra"
    if os.path.exists(hydra_dir):
        target_hydra = f"{run_output_dir}/.hydra"
        shutil.move(hydra_dir, target_hydra)
        print(f"Moved .hydra/ to {target_hydra}")

    writer = None
    log_dir = "runs/"  # default
    if cfg.mode == "train":
        writer = SummaryWriter(log_dir=f"{run_output_dir}tensorboard/")
        log_dir = f"{run_output_dir}tensorboard/"
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
        cfg=cfg.logging,  # Pass logging config
    )
    logger.print_config(OmegaConf.to_yaml(cfg, resolve=True))

    # Build model
    unet = build_model(cfg).to(device)

    # Multi-GPU: Wrap UNet BEFORE building diffusion
    # This ensures gradients flow correctly to all GPUs, especially for OpenAI adapter
    gpu_ids = _parse_multi_gpu_flag(cfg.training.multi_gpu)
    if gpu_ids:
        visible = torch.cuda.device_count()
        if max(gpu_ids) >= visible:
            raise ValueError(f"Requested GPU id {max(gpu_ids)} but only {visible} GPUs visible.")
        print(f"Using GPUs {gpu_ids} with torch.nn.DataParallel")
        unet = torch.nn.DataParallel(unet, device_ids=gpu_ids).cuda(gpu_ids[0])
        print(f"  Wrapped UNet in DataParallel (primary device: cuda:{gpu_ids[0]})")

    # Build diffusion with potentially wrapped UNet
    diffusion = Diffusion.build_diffusion(unet, cfg, device)

    # Ensure diffusion is on correct device
    if not gpu_ids:
        diffusion = diffusion.to(device)
    else:
        # Already on correct device via UNet wrapping
        print(f"  Diffusion built with multi-GPU UNet")

    # Get dataloaders
    dataloaders = get_dataloaders(cfg)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']
    sample_dl = dataloaders['sample']

    if cfg.dataset.name == 'isles24':
        assert cfg.model.image_channels == len(cfg.dataset.modalities), "Model image_channels must match number of modalities"

    if cfg.mode == "train":
        optimizer, scheduler = get_optimizer_and_scheduler(cfg, diffusion)
        # Deprecated: Legacy epoch-based training - to be removed after step-based confirmation
        # train_losses, test_losses, best_epoch = train_and_evaluate(
        #     cfg,
        #     diffusion,
        #     train_dl,
        #     test_dl,
        #     optimizer,
        #     scheduler,
        #     logger=logger,
        #     model_save_path_template=model_save_path_template,
        # )
        # plot_losses(train_losses, test_losses)

        # New step-based training
        step_based_train(
            cfg,
            diffusion,
            dataloaders,
            optimizer,
            scheduler,
            logger,
            run_dir=run_output_dir  # Pass run_dir
        )
    elif cfg.mode == "evaluate":
        # Stub for evaluation: Load model/EMA from config and visualize
        # Assuming load_checkpoint function exists or add a simple loader
        checkpoint_path = f"{cfg.training.evaluation.run_dir}{cfg.training.checkpoint_path_template.format(cfg.training.evaluation.step)}"
        if cfg.training.evaluation.ema_rate is not None:
            checkpoint_path = checkpoint_path.replace('.pth', f'_ema_{cfg.training.evaluation.ema_rate}_{cfg.training.evaluation.step:06d}.pth')
        diffusion.load_state_dict(torch.load(checkpoint_path))
        visualize_best_model_predictions(diffusion, sample_dl.dataset, cfg)
    logger.close()

if __name__ == '__main__':
    main()