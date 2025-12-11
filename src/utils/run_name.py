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
        str: Formatted loss string (e.g., 'lMSE', 'lMSE_dw10_d05_b05_cal100_w25h')
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
        
        # Calibration (only if enabled and weight > 0)
        calibration = aux.get('calibration', {})
        if calibration.get('enabled', False) and calibration.get('weight', 0) > 0:
            parts.append(f"cal{format_loss_weight(calibration['weight'])}")
        
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

def generate_augmentation_string(aug_cfg):
    """
    Generate compact augmentation string for run name.
    
    Encodes augmentation strategy into a short identifier that fits naturally
    into the run name structure. Uses Hydra's _name_ metadata when available
    to identify preset configs.
    
    Args:
        aug_cfg: OmegaConf config from configs/augmentation/, or None
    
    Returns:
        String encoding of augmentation strategy:
        - "augNONE" - No augmentation or all transforms disabled
        - "augLIGHT2D" - Light preset (configs/augmentation/light_2d.yaml)
        - "augAGG2D" - Aggressive preset (configs/augmentation/aggressive_2d.yaml)
        - "augCUSTOM" - Custom user-defined config
    
    Examples:
        >>> cfg_none = OmegaConf.create({'spatial': {'enabled': False}, 'intensity': {'enabled': False}})
        >>> generate_augmentation_string(cfg_none)
        'augNONE'
        
        >>> cfg_none = None
        >>> generate_augmentation_string(cfg_none)
        'augNONE'
    """
    if aug_cfg is None:
        return "augNONE"
    
    # Handle dict vs DictConfig
    if isinstance(aug_cfg, dict):
        spatial_enabled = aug_cfg.get('spatial', {}).get('enabled', False)
        intensity_enabled = aug_cfg.get('intensity', {}).get('enabled', False)
        config_name = aug_cfg.get('_name_', '').lower()
    else:
        spatial_enabled = aug_cfg.spatial.enabled
        intensity_enabled = aug_cfg.intensity.enabled
        config_name = aug_cfg.get('_name_', '').lower()
    
    if not spatial_enabled and not intensity_enabled:
        return "augNONE"
    
    if 'light' in config_name:
        return "augLIGHT2D"
    elif 'aggressive' in config_name or 'agg' in config_name:
        return "augAGG2D"
    elif config_name == 'none':
        return "augNONE"
    else:
        # Custom config or unrecognized preset
        return "augCUSTOM"

def generate_amp_string(cfg):
    """Generate AMP string for run name.
    
    Only adds to run name when AMP is enabled with FP16/BF16.
    Returns empty string for FP32 (default) to keep run names clean.
    
    Args:
        cfg: Configuration object (dict or DictConfig)
    
    Returns:
        str: AMP string for run name, or empty string if disabled/FP32
        
    Examples:
        amp.enabled=false → "" (no string added)
        amp.enabled=true, dtype=float32 → "" (effectively disabled)
        amp.enabled=true, dtype=float16 → "ampFP16"
        amp.enabled=true, dtype=bfloat16 → "ampBF16"
    """
    # Access AMP config
    if isinstance(cfg, dict):
        amp_cfg = cfg.get('training', {}).get('amp', {})
    else:
        amp_cfg = cfg.training.get('amp', {}) if hasattr(cfg, 'training') else {}
    
    enabled = amp_cfg.get('enabled', False)
    dtype = amp_cfg.get('dtype', 'float32')
    
    # No AMP string when disabled or using FP32 (keeps backward compat)
    if not enabled or dtype == 'float32':
        return ""
    
    # Map dtype to compact string
    dtype_map = {
        'float16': 'ampFP16',
        'bfloat16': 'ampBF16',
    }
    return dtype_map.get(dtype, "")


def generate_batch_string(cfg):
    """Generate batch string with optional accumulation encoding.
    
    Encodes both physical batch size and accumulation steps into run name.
    This provides transparency about memory requirements (physical batch)
    and training strategy (accumulation).
    
    Args:
        cfg: Configuration object (dict or DictConfig)
    
    Returns:
        str: Batch string for run name
        
    Examples:
        accumulation_steps=None → "b4" (no accumulation)
        accumulation_steps=1 → "b4" (no accumulation, explicit)
        accumulation_steps=4 → "b4x4" (physical=4, effective=16)
        accumulation_steps=8 → "b4x8" (physical=4, effective=32)
        
    Rationale:
        - b{physical} alone: No accumulation (clean format)
        - b{physical}x{accumulation}: Shows both memory footprint and training strategy
        - Makes accumulation visible in run names for debugging and comparison
        - Distinguishes b4x4 (with accumulation) from b16x1 (without) even though both are effective batch 16
    """
    # Access config (fail-fast, no defaults)
    if isinstance(cfg, dict):
        batch_size = cfg['environment']['dataset']['train_batch_size']
        accum = cfg['training']['gradient']['accumulation_steps']
    else:
        batch_size = cfg.environment.dataset.train_batch_size
        accum = cfg.training.gradient.accumulation_steps
    
    # None or 1 means no accumulation (show clean format)
    if accum is None or accum == 1:
        return f"b{batch_size}"
    else:
        return f"b{batch_size}x{accum}"

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
    
    # Model string - architecture specific
    architecture = model['architecture']
    
    if architecture == 'org_medsegdiff':
        # ORGMedSegDiff: {arch}_{version}_{size}_{channels}c_{channel_mult}_{res_blocks}rb_{heads}h[_hwy|_nohwy]
        # Example: org_medsegdiff_new_256_128c_1-2-4-8_2rb_4h_hwy (S variant with highway)
        #          org_medsegdiff_new_256_128c_1-2-4-8_2rb_4h_nohwy (S variant without highway)
        #          org_medsegdiff_new_256_128c_1-2-4-4-8_2rb_4h_hwy (B variant)
        #          org_medsegdiff_new_256_128c_1-1-2-2-4-4_2rb_4h_hwy (L variant)
        version = model.get('version', 'new')
        channel_mult = model.get('channel_mult', '').replace(',', '-')
        
        # Highway network indicator
        highway_cfg = model.get('highway', {})
        if isinstance(highway_cfg, dict):
            highway_enabled = highway_cfg.get('enabled', True)
        else:
            highway_enabled = highway_cfg.enabled if hasattr(highway_cfg, 'enabled') else True
        highway_str = "hwy" if highway_enabled else "nohwy"
        
        model_str = (f"{architecture}_{version}_{model['image_size']}_"
                     f"{model['model_channels']}c_{channel_mult}_"
                     f"{model.get('num_res_blocks', 2)}rb_{model.get('num_heads', 4)}h_{highway_str}")
    elif architecture == 'diffswintr':
        # DiffSwinTr: diffswintr_{size}_{embed_dim}d_{depths}_{num_heads}h_{window_size}w[_cem]
        # Example: diffswintr_256_64d_2-2-2-2_2-4-8-16h_8w_cem (Small with CEM)
        #          diffswintr_256_96d_2-2-6-2_3-6-12-24h_8w_cem (Base with CEM)
        #          diffswintr_256_128d_2-2-18-2_4-8-16-32h_8w (Large without CEM)
        
        # Format depths as dash-separated string
        depths = model.get('depths', [])
        if isinstance(depths, (list, tuple)):
            depths_str = '-'.join(str(d) for d in depths)
        else:
            depths_str = str(depths).replace(',', '-')
        
        # Format num_heads as dash-separated string
        num_heads = model.get('num_heads', [])
        if isinstance(num_heads, (list, tuple)):
            heads_str = '-'.join(str(h) for h in num_heads)
        else:
            heads_str = str(num_heads).replace(',', '-')
        
        model_str = (f"{architecture}_{model['image_size']}_"
                     f"{model['embed_dim']}d_{depths_str}_{heads_str}h_{model['window_size']}w")
        
        # Add CEM suffix if enabled
        if model.get('cem_enabled', True):
            model_str += "_cem"
    else:
        # MedSegDiff (default): {architecture}_{image_size}_{num_layers}l_{first_conv_channels}c_{att_heads}x{att_head_dim}a_{time_embedding_dim}t_{bottleneck_transformer_layers}btl
        model_str = (f"{architecture}_{model['image_size']}_{model['num_layers']}l_{model['first_conv_channels']}c_"
                     f"{model['att_heads']}x{model['att_head_dim']}a_{model['time_embedding_dim']}t_"
                     f"{model['bottleneck_transformer_layers']}btl")
    
    # Batch string (with accumulation encoding if enabled)
    batch_str = generate_batch_string(cfg)
    
    # AMP string (only added if enabled with FP16/BF16)
    amp_str = generate_amp_string(cfg)
    
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
    
    # NEW: Add augmentation string
    aug_cfg = cfg.get('augmentation', None) if isinstance(cfg, dict) else (cfg.augmentation if hasattr(cfg, 'augmentation') else None)
    aug_str = generate_augmentation_string(aug_cfg)
    
    # Combine all parts with separated optimizer and scheduler
    # Insert aug_str between loss_str and diffusion_str
    # Insert amp_str after batch_str (only if non-empty)
    parts = [model_str, batch_str]
    if amp_str:  # Only add AMP string if enabled (FP16/BF16)
        parts.append(amp_str)
    parts.extend([optimizer_str, scheduler_str, steps_str, loss_str, aug_str, diffusion_str, timestamp])
    
    run_name = '_'.join(parts)
    
    return run_name
