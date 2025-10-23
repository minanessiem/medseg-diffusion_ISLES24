import datetime

def generate_run_name(cfg, timestamp: str = None) -> str:
    """Generate the run name string based on the resolved config.

    Args:
        cfg: The resolved config (DictConfig or plain dict).
        timestamp: Optional timestamp string; if None, generate current timestamp.

    Returns:
        str: The formatted run name.
    """
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    scheduler_str = ""
    if isinstance(cfg, dict):
        training = cfg.get('training', {})
        model = cfg.get('model', {})
        dataset = cfg.get('dataset', {})
    else:  # Assume DictConfig
        training = cfg.training
        model = cfg.model
        dataset = cfg.dataset
    
    if training.get('scheduler_type') == 'reduce_lr':
        params = {
            'rlrfctr': 'reduce_lr_factor',
            'rlrpat': 'reduce_lr_patience',
            'rlrthrsh': 'reduce_lr_threshold',
            'rlrcool': 'reduce_lr_cooldown'
        }
        for abbr, key in params.items():
            val = training.get(key)
            if val is not None:
                scheduler_str += f"_{abbr}{val}"
    
    run_name = (f"unet_img{model['image_size']}_numlayers{model['num_layers']}_firstconv{model['first_conv_channels']}_"
                f"timembdim{model['time_embedding_dim']}_attheads{model['att_heads']}_attheaddim{model['att_head_dim']}_"
                f"btllayers{model['bottleneck_transformer_layers']}_btchsz{dataset['train_batch_size']}_"
                f"lr{training['learning_rate']}_maxsteps{training['max_steps']}_diffsteps{training['timesteps']}"
                f"{scheduler_str}_{timestamp}")
    
    return run_name
