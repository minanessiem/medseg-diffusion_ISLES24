import datetime

def generate_run_name(cfg, timestamp: str = None) -> str:
    """Generate the run name string based on the resolved Hydra config.

    Args:
        cfg: The resolved DictConfig object.
        timestamp: Optional timestamp string; if None, generate current timestamp.

    Returns:
        str: The formatted run name.
    """
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    scheduler_str = ""
    if cfg.training.get('scheduler_type') == 'reduce_lr':
        params = {
            'rlrfctr': 'reduce_lr_factor',
            'rlrpat': 'reduce_lr_patience',
            'rlrthrsh': 'reduce_lr_threshold',
            'rlrcool': 'reduce_lr_cooldown'
        }
        for abbr, key in params.items():
            val = cfg.training.get(key)
            if val is not None:
                scheduler_str += f"_{abbr}{val}"
    
    run_name = (f"unet_img{cfg.model.image_size}_numlayers{cfg.model.num_layers}_firstconv{cfg.model.first_conv_channels}_"
                f"timembdim{cfg.model.time_embedding_dim}_attheads{cfg.model.att_heads}_attheaddim{cfg.model.att_head_dim}_"
                f"btllayers{cfg.model.bottleneck_transformer_layers}_btchsz{cfg.dataset.train_batch_size}_"
                f"lr{cfg.training.learning_rate}_maxsteps{cfg.training.max_steps}_diffsteps{cfg.training.timesteps}"
                f"{scheduler_str}_{timestamp}")
    
    return run_name
