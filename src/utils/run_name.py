import datetime

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
        diffusion = cfg.get('diffusion', {})
        loss = cfg.get('loss', {})
    else:  # Assume DictConfig
        training = cfg.training
        model = cfg.model
        dataset = cfg.dataset
        optimizer = cfg.optimizer
        diffusion = cfg.diffusion
        loss = cfg.loss if hasattr(cfg, 'loss') else {}
    
    # Model string: {architecture}_{image_size}_{num_layers}l_{first_conv_channels}c_{att_heads}x{att_head_dim}a_{time_embedding_dim}t_{bottleneck_transformer_layers}btl
    model_str = (f"{model['architecture']}_{model['image_size']}_{model['num_layers']}l_{model['first_conv_channels']}c_"
                 f"{model['att_heads']}x{model['att_head_dim']}a_{model['time_embedding_dim']}t_"
                 f"{model['bottleneck_transformer_layers']}btl")
    
    # Batch size
    batch_str = f"b{dataset['train_batch_size']}"
    
    # Optimizer string: {lr_formatted}lr_{scheduler_type}s_{factor}f_{patience}p_{threshold}t_{cooldown}c_{interval}i
    lr = float(optimizer['learning_rate'])  # Convert to float in case it's a string
    lr_formatted = f"{lr:.0e}".replace('e-0', 'e').replace('e-', 'e')  # 2e-4 -> 2e4
    optimizer_str = f"{lr_formatted}lr"
    
    if optimizer.get('scheduler_type') == 'reduce_lr':
        optimizer_str += "_reducelrs"
        
        # Factor: 0.75 -> 075
        factor = optimizer.get('reduce_lr_factor')
        if factor is not None:
            factor_str = f"{factor:.2f}".replace('.', '')  # Remove decimal point
            optimizer_str += f"_{factor_str}f"
        
        # Patience
        patience = optimizer.get('reduce_lr_patience')
        if patience is not None:
            optimizer_str += f"_{patience}p"
        
        # Threshold: 1e-4 -> 1e4
        threshold = optimizer.get('reduce_lr_threshold')
        if threshold is not None:
            threshold = float(threshold)  # Convert to float in case it's a string
            threshold_str = f"{threshold:.0e}".replace('e-0', 'e').replace('e-', 'e')
            optimizer_str += f"_{threshold_str}t"
        
        # Cooldown
        cooldown = optimizer.get('reduce_lr_cooldown')
        if cooldown is not None:
            optimizer_str += f"_{cooldown}c"
        
        # Interval: 1000 -> 1Ki
        interval = optimizer.get('scheduler_interval')
        if interval is not None:
            if interval >= 1000 and interval % 1000 == 0:
                interval_str = f"{interval // 1000}Ki"
            else:
                interval_str = f"{interval}i"
            optimizer_str += f"_{interval_str}"
    
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
    
    # Combine all parts (loss comes after training steps, before diffusion)
    run_name = f"{model_str}_{batch_str}_{optimizer_str}_{steps_str}_{loss_str}_{diffusion_str}_{timestamp}"
    
    return run_name
