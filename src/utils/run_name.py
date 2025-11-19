import datetime

def format_learning_rate(lr):
    """Format learning rate for run name.
    
    Args:
        lr: Float learning rate (e.g., 1e-4, 2e-4)
        
    Returns:
        str: Formatted LR string (e.g., '1e4', '2e4', '5e5')
    
    Examples:
        1e-4 -> '1e4'
        2e-4 -> '2e4'
        5e-5 -> '5e5'
    """
    lr = float(lr)  # Convert to float in case it's a string
    lr_str = f"{lr:.0e}".replace('e-0', 'e').replace('e-', 'e')
    return lr_str

def format_loss_warmup(steps):
    """Format warmup steps following project conventions.
    
    Args:
        steps: Integer warmup steps
        
    Returns:
        str: Formatted warmup string (e.g., 0, 25h, 5K)
    """
    if steps == 0:
        return "0"
    if steps >= 1000 and steps % 1000 == 0:
        return f"{steps // 1000}K"
    if steps >= 100 and steps % 100 == 0:
        return f"{steps // 100}h"
    return str(steps)

def format_loss_weight(weight):
    """Format loss weight as 2-digit string.
    
    Args:
        weight: Float weight value (e.g., 0.5, 1.0)
        
    Returns:
        str: 2-digit string (e.g., 0.5 -> '05', 1.0 -> '10')
    """
    return f"{int(weight * 100):02d}"

def generate_loss_string(loss_cfg):
    """Generate loss substring for run name.
    
    Args:
        loss_cfg: Loss configuration dict or DictConfig
        
    Returns:
        str: Formatted loss string (e.g., 'lMSE', 'lMSE_dw10_d05_b05_w25h')
    """
    # Handle both dict and DictConfig
    if hasattr(loss_cfg, 'loss_type'):
        loss_type = loss_cfg.loss_type
        aux = loss_cfg.get('auxiliary_losses', {})
    else:
        loss_type = loss_cfg.get('loss_type', 'MSE')
        aux = loss_cfg.get('auxiliary_losses', {})
    
    parts = [f"l{loss_type}"]
    
    # Only add auxiliary loss details if enabled
    if aux.get('enabled', False):
        # Diffusion weight
        dw = aux.get('diffusion_weight', 1.0)
        parts.append(f"dw{format_loss_weight(dw)}")
        
        # Dice (only if enabled and weight > 0)
        dice = aux.get('dice', {})
        if dice.get('enabled', False) and dice.get('weight', 0) > 0:
            parts.append(f"d{format_loss_weight(dice['weight'])}")
        
        # BCE (only if enabled and weight > 0)
        bce = aux.get('bce', {})
        if bce.get('enabled', False) and bce.get('weight', 0) > 0:
            parts.append(f"b{format_loss_weight(bce['weight'])}")
        
        # Warmup
        warmup = aux.get('warmup_steps', 0)
        parts.append(f"w{format_loss_warmup(warmup)}")
    
    return '_'.join(parts)

def generate_optimizer_string(opt_cfg):
    """Generate optimizer string from separated config.
    
    Args:
        opt_cfg: Optimizer configuration (DictConfig or dict)
    
    Returns:
        str: Compact optimizer string for run name
    
    Raises:
        KeyError: If required config keys are missing (fail-fast, no defaults)
    
    Examples:
        adamw_1e4lr_wd00 → "adamw1e4_wd00"
        adamw_2e4lr_wd00 → "adamw2e4_wd00"
        adamw_1e4lr_wd01 → "adamw1e4_wd01"
        adam_2e4lr → "adam2e4"
    """
    # Handle both dict and DictConfig (no defaults - fail fast)
    if hasattr(opt_cfg, 'optimizer_class'):
        opt_class = opt_cfg.optimizer_class
        lr = opt_cfg.learning_rate
        wd = opt_cfg.weight_decay
    else:
        opt_class = opt_cfg['optimizer_class']  # Will raise KeyError if missing
        lr = opt_cfg['learning_rate']  # Will raise KeyError if missing
        wd = opt_cfg['weight_decay']  # Will raise KeyError if missing
    
    lr_str = format_learning_rate(lr)
    parts = [f"{opt_class}{lr_str}"]
    
    # Add weight decay if non-zero
    if wd > 0:
        wd_str = f"wd{int(wd * 100):02d}"  # 0.01 → "wd01", 0.1 → "wd10"
        parts.append(wd_str)
    else:
        # Always show wd00 for clarity (distinguishes from old format)
        parts.append("wd00")
    
    return '_'.join(parts)

def generate_scheduler_string(sched_cfg):
    """Generate scheduler string from config.
    
    Args:
        sched_cfg: Scheduler configuration (DictConfig or dict)
    
    Returns:
        str: Compact scheduler string for run name
    
    Raises:
        KeyError: If required config keys are missing (fail-fast, no defaults)
    
    Examples:
        warmup_cosine_10pct → "wcos10"
        warmup_constant_10pct → "wcon10"
        cosine → "cos"
        reduce_lr (factor=0.75, patience=2) → "rlr75f_2p"
        constant → "const"
    """
    # Handle both dict and DictConfig (no defaults - fail fast)
    if hasattr(sched_cfg, 'scheduler_type'):
        sched_type = sched_cfg.scheduler_type
    else:
        sched_type = sched_cfg['scheduler_type']  # Will raise KeyError if missing
    
    if sched_type == 'warmup_cosine':
        # Get warmup fraction (required)
        if hasattr(sched_cfg, 'warmup_fraction'):
            warmup_frac = sched_cfg.warmup_fraction
        else:
            warmup_frac = sched_cfg['warmup_fraction']  # Will raise KeyError if missing
        warmup_pct = int(warmup_frac * 100)
        return f"wcos{warmup_pct}"
    
    elif sched_type == 'warmup_constant':
        # Get warmup fraction (required)
        if hasattr(sched_cfg, 'warmup_fraction'):
            warmup_frac = sched_cfg.warmup_fraction
        else:
            warmup_frac = sched_cfg['warmup_fraction']  # Will raise KeyError if missing
        warmup_pct = int(warmup_frac * 100)
        return f"wcon{warmup_pct}"
    
    elif sched_type == 'cosine':
        return "cos"
    
    elif sched_type == 'reduce_lr':
        # Get factor and patience (required)
        if hasattr(sched_cfg, 'factor'):
            factor = sched_cfg.factor
            patience = sched_cfg.patience
        else:
            factor = sched_cfg['factor']  # Will raise KeyError if missing
            patience = sched_cfg['patience']  # Will raise KeyError if missing
        
        factor_str = f"{int(factor * 100):02d}"  # 0.75 → "75", 0.5 → "50"
        return f"rlr{factor_str}f_{patience}p"
    
    elif sched_type == 'constant':
        return "const"
    
    else:
        # Fallback: first 4 chars
        return sched_type[:4]

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
    
    if isinstance(cfg, dict):
        training = cfg.get('training', {})
        model = cfg.get('model', {})
        dataset = cfg.get('dataset', {})
        optimizer = cfg.get('optimizer', {})
        scheduler = cfg.get('scheduler', {})
        diffusion = cfg.get('diffusion', {})
        loss = cfg.get('loss', {})
    else:  # Assume DictConfig
        training = cfg.training
        model = cfg.model
        dataset = cfg.dataset
        optimizer = cfg.optimizer
        scheduler = cfg.scheduler
        diffusion = cfg.diffusion
        loss = cfg.loss if hasattr(cfg, 'loss') else {}
    
    # Model string: {architecture}_{image_size}_{num_layers}l_{first_conv_channels}c_{att_heads}x{att_head_dim}a_{time_embedding_dim}t_{bottleneck_transformer_layers}btl
    model_str = (f"{model['architecture']}_{model['image_size']}_{model['num_layers']}l_{model['first_conv_channels']}c_"
                 f"{model['att_heads']}x{model['att_head_dim']}a_{model['time_embedding_dim']}t_"
                 f"{model['bottleneck_transformer_layers']}btl")
    
    # Batch size
    batch_str = f"b{dataset['train_batch_size']}"
    
    # Optimizer and scheduler strings (separated)
    optimizer_str = generate_optimizer_string(optimizer)
    scheduler_str = generate_scheduler_string(scheduler)
    
    # Training steps: 100000 -> 100K
    max_steps = training['max_steps']
    if max_steps >= 1000 and max_steps % 1000 == 0:
        steps_str = f"s{max_steps // 1000}K"
    else:
        steps_str = f"s{max_steps}"
    
    # Diffusion string: {oai_prefix}_{sampling_mode}_ds{timesteps}_nz{noise_schedule}_tr{timestep_respacing}
    diffusion_parts = []
    
    # Add "oai" prefix if type is OpenAI_DDPM
    if diffusion.get('type') == 'OpenAI_DDPM':
        diffusion_parts.append('oai')
    
    # Sampling mode
    sampling_mode = diffusion.get('sampling_mode', 'ddpm')
    diffusion_parts.append(sampling_mode)
    
    # Timesteps
    diffusion_parts.append(f"ds{diffusion['timesteps']}")
    
    # Noise schedule
    noise_schedule = diffusion.get('noise_schedule', 'linear')
    diffusion_parts.append(f"nz{noise_schedule}")
    
    # Timestep respacing (only if not empty)
    timestep_respacing = diffusion.get('timestep_respacing', '')
    if timestep_respacing:
        diffusion_parts.append(f"tr{timestep_respacing}")
    
    diffusion_str = '_'.join(diffusion_parts)
    
    # Loss string: l{loss_type}[_dw{diffusion_weight}_d{dice_weight}_b{bce_weight}_w{warmup}]
    loss_str = generate_loss_string(loss)
    
    # Combine all parts with separated optimizer and scheduler
    run_name = f"{model_str}_{batch_str}_{optimizer_str}_{scheduler_str}_{steps_str}_{loss_str}_{diffusion_str}_{timestamp}"
    
    return run_name
