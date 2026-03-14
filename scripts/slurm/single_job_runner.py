#!/usr/bin/env python3

import argparse
from typing import List, Optional
from datetime import datetime
import os
import yaml
from typing import Dict
import pprint

from src.utils.run_name import generate_run_name

from scripts.slurm.base_run_config import BASE_CONFIG, SLURM_TEMPLATE, update_logdir_paths
from scripts.slurm.job_runner import SlurmJobRunner
from scripts.slurm.utils.commandline_utils import add_config_arguments, update_config_from_args

def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
    """Recursively merge dict2 into dict1."""
    for key, value in dict2.items():
        if isinstance(value, dict) and key in dict1 and isinstance(dict1[key], dict):
            dict1[key] = deep_merge(dict1[key], value)
        else:
            dict1[key] = value
    return dict1


def _load_yaml_with_package(config_path: str) -> tuple[Dict, Optional[str]]:
    """
    Load YAML and extract optional Hydra package directive from leading comments.

    Recognized directive format:
      # @package _global_
    """
    with open(config_path, "r", encoding="utf-8") as f:
        raw = f.read()

    package = None
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if stripped.startswith("# @package "):
                package = stripped[len("# @package "):].strip()
            continue
        # Stop scanning once real YAML content starts.
        break

    cfg = yaml.safe_load(raw) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected dict YAML at {config_path}, got {type(cfg)}")
    return cfg, package


def _is_global_package(package: Optional[str]) -> bool:
    return package == "_global_"


def _normalize_default_section_key(raw_key: str) -> tuple[str, bool]:
    """
    Normalize Hydra defaults key syntax.

    Supports:
      - /section
      - section
      - override /section

    Rejects non-Hydra-compliant override syntax (e.g. "override section").
    """
    key = str(raw_key).strip()
    if key.startswith("override "):
        if not key.startswith("override /"):
            raise ValueError(
                f"Invalid defaults override syntax '{raw_key}'. "
                "Expected 'override /<section>: <name>'."
            )
        section = key[len("override /"):].strip()
        if not section:
            raise ValueError(f"Invalid defaults key '{raw_key}': empty override section.")
        return section, True

    if key.startswith("/"):
        section = key[1:].strip()
    else:
        section = key

    if not section:
        raise ValueError(f"Invalid defaults key '{raw_key}': empty section.")
    return section, False


def apply_override(cfg: Dict, override: str) -> Dict:
    """Apply a single key=value override, supporting dotted paths and config group references."""
    if '=' not in override:
        return cfg
    key_path, value = override.split('=', 1)
    keys = key_path.split('.')
    
    # Special handling for config group references (e.g., loss=mse_loss_only)
    # These should load the corresponding config file, not assign as string
    if len(keys) == 1:  # Simple key (no dots) - might be a config group
        config_group = keys[0]
        potential_file = f"configs/{config_group}/{value}.yaml"
        
        if os.path.exists(potential_file):
            # Use load_group_config to recursively resolve all defaults
            group_cfg, group_package = load_group_config(config_group, value)

            # Merge into main config
            if _is_global_package(group_package):
                cfg = deep_merge(cfg, group_cfg)
            else:
                cfg[config_group] = group_cfg
            return cfg
    
    # Standard dotted path override (existing logic)
    current = cfg
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Parse value: handle Hydra list syntax [val1,val2,...] and primitives
    if value.startswith('[') and value.endswith(']'):
        # Hydra list syntax: [0,1,2,3] or [true,false]
        list_content = value[1:-1].strip()
        if list_content:
            items = [item.strip() for item in list_content.split(',')]
            parsed_items = []
            for item in items:
                # Try to parse each item as int/float/bool
                try:
                    parsed_items.append(int(item))
                except ValueError:
                    try:
                        parsed_items.append(float(item))
                    except ValueError:
                        if item.lower() == 'true':
                            parsed_items.append(True)
                        elif item.lower() == 'false':
                            parsed_items.append(False)
                        else:
                            parsed_items.append(item)  # Keep as string
            value = parsed_items
        else:
            value = []  # Empty list
    else:
        # Try to convert single value to int/float/bool
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                # else: Keep as string
    
    current[keys[-1]] = value
    return cfg

def resolve_interpolations(cfg: Dict, root: Dict = None) -> Dict:
    """Recursively resolve ${path.to.key} interpolations in the dict."""
    if root is None:
        root = cfg
    if isinstance(cfg, dict):
        for key, value in cfg.items():
            cfg[key] = resolve_interpolations(value, root)
        return cfg
    elif isinstance(cfg, str) and cfg.startswith('${') and cfg.endswith('}'):
        path = cfg[2:-1].split('.')
        current = root
        for p in path:
            if isinstance(current, dict) and p in current:
                current = current[p]
            else:
                raise KeyError(f"Interpolation key '{cfg}' not found")
        return current
    return cfg

def _find_self_index(defaults: list) -> int:
    """
    Find the position of _self_ in a defaults list.
    
    Returns the index of _self_, or len(defaults) if not present
    (Hydra treats absent _self_ as if it were last).
    """
    for i, default in enumerate(defaults):
        if default == '_self_':
            return i
    return len(defaults)


def _process_group_default(default, group: str) -> tuple:
    """
    Process a single default entry from a group config's defaults list.
    
    Args:
        default: Either a string (same-group reference) or dict (cross-group)
        group: The current config group directory
        
    Returns:
        Tuple of (section_key, loaded_config, package) where section_key is None
        for same-group merges or the section name for cross-group merges
    """
    if isinstance(default, str):
        sub_cfg, package = load_group_config(group, default)
        return None, sub_cfg, package
    elif isinstance(default, dict):
        key, file_name = next(iter(default.items()))
        section, _ = _normalize_default_section_key(key)
        sub_cfg, package = load_group_config(section, file_name)
        return section, sub_cfg, package
    return None, {}, None


def load_group_config(group: str, name: str) -> tuple[Dict, Optional[str]]:
    """
    Load a config file from a config group, recursively resolving its defaults.
    
    Respects _self_ position for merge ordering (Hydra semantics):
    - Defaults BEFORE _self_: overridden by local values
    - Defaults AFTER _self_: override local values
    - If _self_ is absent: treated as last (local values win)
    
    Args:
        group: Config group directory (e.g., 'training', 'loss', 'model')
        name: Config file name without .yaml extension
        
    Returns:
        Tuple of (fully resolved config dict, package directive if present)
    """
    config_path = f"configs/{group}/{name}.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    cfg, package = _load_yaml_with_package(config_path)
    
    # Recursively resolve defaults within this config
    if 'defaults' in cfg:
        defaults = cfg.pop('defaults')
        local_values = dict(cfg)
        
        # Find _self_ position to determine merge order
        self_idx = _find_self_index(defaults)
        before_self = [d for d in defaults[:self_idx] if d != '_self_']
        after_self = [d for d in defaults[self_idx + 1:] if d != '_self_']
        
        # Step 1: Merge defaults BEFORE _self_ (local values will override these)
        cfg = {}
        for default in before_self:
            section, sub_cfg, sub_package = _process_group_default(default, group)
            if section is not None and not _is_global_package(sub_package):
                cfg[section] = sub_cfg
            else:
                cfg = deep_merge(cfg, sub_cfg)
        
        # Step 2: Apply _self_ (local values override defaults before it)
        cfg = deep_merge(cfg, local_values)
        
        # Step 3: Merge defaults AFTER _self_ (these override local values)
        for default in after_self:
            section, sub_cfg, sub_package = _process_group_default(default, group)
            if section is not None and not _is_global_package(sub_package):
                cfg[section] = sub_cfg
            else:
                cfg = deep_merge(cfg, sub_cfg)
    
    return cfg, package


def _process_top_level_default(default, current_cfg: Dict) -> Dict:
    """
    Process a single default entry from a top-level config's defaults list.
    
    Handles string references (recursive config loads), dict references
    (config group loads), and Hydra's 'override /' syntax.
    
    Args:
        default: Either a string or dict default entry
        current_cfg: Current accumulated config (for override vs merge logic)
        
    Returns:
        Updated config dict with the default merged in
    """
    cfg = current_cfg
    
    if isinstance(default, str):
        # Simple string reference: load another config file recursively
        # Keep interpolations unresolved at nested levels; resolve once at top-level.
        sub_cfg = load_config(default, [], resolve_final=False)
        cfg = deep_merge(cfg, sub_cfg)
    elif isinstance(default, dict):
        key, file_name = next(iter(default.items()))
        
        # Handle Hydra defaults key syntax (strict, no non-Hydra tolerance)
        section, is_override = _normalize_default_section_key(key)
        
        # Use load_group_config to recursively resolve defaults
        sub_cfg, package = load_group_config(section, file_name)
        
        # Global-packaged groups merge at root; otherwise section assignment/merge.
        if _is_global_package(package):
            cfg = deep_merge(cfg, sub_cfg)
        elif is_override:
            cfg[section] = sub_cfg
        else:
            if section in cfg and isinstance(cfg[section], dict):
                cfg[section] = deep_merge(cfg[section], sub_cfg)
            else:
                cfg[section] = sub_cfg
    
    return cfg


def load_config(config_name: str, overrides: List[str], resolve_final: bool = True) -> Dict:
    """
    Load and merge YAML configs mimicking Hydra composition.
    
    Respects _self_ position for merge ordering:
    - Defaults BEFORE _self_: overridden by local values
    - Defaults AFTER _self_: override local values
    - If _self_ is absent: treated as last (local values win)
    """
    config_path = f"configs/{config_name}.yaml"  # Relative from project root
    cfg, _ = _load_yaml_with_package(config_path)
    
    # Merge defaults with nesting (recursively)
    if 'defaults' in cfg:
        defaults = cfg.pop('defaults')
        local_values = {k: v for k, v in cfg.items()}
        
        # Find _self_ position to determine merge order
        self_idx = _find_self_index(defaults)
        before_self = [d for d in defaults[:self_idx] if d != '_self_']
        after_self = [d for d in defaults[self_idx + 1:] if d != '_self_']
        
        # Step 1: Merge defaults BEFORE _self_ (local values will override these)
        cfg = {}
        for default in before_self:
            cfg = _process_top_level_default(default, cfg)
        
        # Step 2: Apply _self_ (local values override defaults before it)
        cfg = deep_merge(cfg, local_values)
        
        # Step 3: Merge defaults AFTER _self_ (these override local values)
        for default in after_self:
            cfg = _process_top_level_default(default, cfg)
    
    # Apply overrides (command-line overrides come last, always win)
    for ovr in overrides:
        cfg = apply_override(cfg, ovr)
    
    # Resolve interpolations only once at the final top-level compose stage.
    # Nested load_config() calls keep ${...} placeholders so later overrides
    # (e.g., environment swaps) still influence referenced values.
    if resolve_final:
        cfg = resolve_interpolations(cfg)
    
    return cfg


def _get_override_value(overrides: List[str], key: str) -> Optional[str]:
    """Return the last override value for an exact top-level key (e.g., distribution)."""
    value = None
    for override in overrides:
        cleaned = override.lstrip("+~")
        if "=" not in cleaned:
            continue
        override_key, override_value = cleaned.split("=", 1)
        if override_key == key:
            value = override_value
    return value


def _resolve_resume_strategy(resume_dir: str, overrides: List[str], config: Dict) -> str:
    """
    Resolve launch strategy for resume jobs.

    Resolution order:
      1) Explicit CLI override: distribution=... or distribution.strategy=...
      2) Stored Hydra config at <run_dir>/.hydra/config.yaml (container path or mapped host path)

    Fails fast if strategy cannot be resolved, to avoid silently emitting a DP launcher.
    """
    override_strategy = _get_override_value(overrides, "distribution")
    if override_strategy is None:
        override_strategy = _get_override_value(overrides, "distribution.strategy")
    if override_strategy is not None:
        override_strategy = str(override_strategy).strip().lower()
        if override_strategy in {"dp", "ddp"}:
            return override_strategy
        raise ValueError(
            f"Unsupported distribution override '{override_strategy}'. "
            "Expected one of: dp, ddp."
        )

    resume_dir = resume_dir.rstrip("/")
    candidate_resume_dirs = [resume_dir]

    # Map container resume path to host path so strategy detection works on host
    # before container launch.
    container_outputs_base = str(config.get("container_outputs_base", "")).rstrip("/")
    host_outputs_dir = str(config.get("host_outputs_dir", ""))
    if container_outputs_base and host_outputs_dir and resume_dir.startswith(container_outputs_base):
        relative = resume_dir[len(container_outputs_base):].lstrip("/")
        candidate_resume_dirs.append(os.path.join(host_outputs_dir, relative))

    for candidate in candidate_resume_dirs:
        hydra_cfg_path = os.path.join(candidate, ".hydra", "config.yaml")
        if not os.path.exists(hydra_cfg_path):
            continue

        with open(hydra_cfg_path, "r") as f:
            stored_cfg = yaml.safe_load(f) or {}

        strategy = (
            stored_cfg.get("distribution", {}).get("strategy")
            if isinstance(stored_cfg, dict)
            else None
        )
        strategy = str(strategy).strip().lower() if strategy is not None else None
        if strategy in {"dp", "ddp"}:
            return strategy

    raise RuntimeError(
        "Could not resolve resume distribution strategy. "
        "Pass an explicit override (distribution=dp|ddp), or ensure "
        "<run_dir>/.hydra/config.yaml is readable from host path mapping."
    )


def _build_training_command(
    strategy: str,
    gpus: int,
    script_name: str,
    script_args: str,
) -> str:
    """Build python or torchrun command based on selected distribution strategy."""
    script_args = script_args.strip()
    suffix = f" {script_args}" if script_args else ""

    if strategy == "ddp":
        if gpus <= 1:
            raise ValueError(
                f"distribution=ddp requires --gpus > 1 for this launcher, got --gpus {gpus}."
            )
        return (
            f"torchrun --standalone --nnodes=1 --nproc_per_node={gpus} "
            f"{script_name}{suffix}"
        ).strip()

    return f"python3 {script_name}{suffix}".strip()

def main():
    parser = argparse.ArgumentParser(description='Submit a single SLURM job for medseg-diffusion training')
    
    # Hydra-related arguments (reused for custom loader)
    parser.add_argument('--config-name', type=str, default='cluster',
                        help='Config name (loads configs/{config-name}.yaml)')
    parser.add_argument('--overrides', nargs='*', default=[],
                        help='Overrides as key=value pairs (e.g., model.image_size=64)')
    
    # Resume support
    parser.add_argument('--resume-dir', type=str, default=None,
                        help='Container path to run directory to resume (e.g., /mnt/outputs/my_run/). '
                             'When set, uses resume_training.py instead of start_training.py.')
    
    # Other args
    parser.add_argument('--dry-run', action='store_true',
                        help='Print job that would be submitted without submitting')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging (adds debug=true to training overrides)')
    parser.add_argument('--job-name', type=str, default=None,
                        help='Custom job name prefix (otherwise auto-generated)')
    
    # SLURM resource arguments
    parser.add_argument('--gpus', type=int, default=None,
                        help='Number of GPUs to request (overrides BASE_CONFIG)')
    parser.add_argument('--partition', type=str, default=None,
                        help='SLURM partition to use (overrides BASE_CONFIG)')
    parser.add_argument('--cpus-per-task', type=int, default=None,
                        help='CPUs per task (overrides BASE_CONFIG)')
    parser.add_argument('--mem', type=str, default=None,
                        help='Memory allocation (e.g., "256G", overrides BASE_CONFIG)')
    parser.add_argument('--time', type=str, default=None,
                        help='Time limit (e.g., "00:30:00" for 30 min, overrides BASE_CONFIG)')
    
    # Add configuration override arguments from BASE_CONFIG (exclude already-added params)
    excluded_params = {'gpus', 'partition', 'cpus_per_task', 'mem', 'time'}
    filtered_config = {k: v for k, v in BASE_CONFIG.items() if k not in excluded_params}
    add_config_arguments(parser, filtered_config)
    
    args = parser.parse_args()
    
    # Update BASE_CONFIG with command line arguments
    config = update_config_from_args(BASE_CONFIG.copy(), args, update_logdir_paths)
    
    # Override SLURM resource parameters if specified
    if args.gpus is not None:
        config["gpus"] = args.gpus
    if args.partition is not None:
        config["partition"] = args.partition
    if args.cpus_per_task is not None:
        config["cpus_per_task"] = args.cpus_per_task
    if args.mem is not None:
        config["mem"] = args.mem
    if args.time is not None:
        config["time"] = args.time
    
    # Parse overrides
    overrides = list(args.overrides)  # Make a copy to avoid modifying original
    
    # Debug mode: add +debug=true to overrides for training script
    # Note: '+' prefix tells Hydra to add a new key (not override existing)
    if args.debug:
        overrides.append("+debug=true")
        print(f"[DEBUG] Debug mode enabled - added +debug=true to overrides")
    
    def _debug(msg):
        if args.debug:
            print(f"[DEBUG:single_job_runner] {msg}")
    
    _debug("="*60)
    _debug("single_job_runner.py started")
    _debug("="*60)
    _debug(f"config-name: {args.config_name}")
    _debug(f"resume-dir: {args.resume_dir}")
    _debug(f"overrides: {overrides}")
    _debug(f"dry-run: {args.dry_run}")
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _debug(f"timestamp: {timestamp}")
    
    # ==========================================================================
    # Branch: Resume mode vs Fresh training mode
    # ==========================================================================
    if args.resume_dir:
        # ======================================================================
        # RESUME MODE: Use resume_training.py with existing run directory
        # ======================================================================
        _debug("Entering RESUME MODE branch")
        resume_dir = args.resume_dir.rstrip('/')
        _debug(f"resume_dir (stripped): {resume_dir}")
        
        # Extract run_name from the resume directory path
        run_name = os.path.basename(resume_dir)
        _debug(f"run_name extracted: {run_name}")
        
        # Set job_name (same as original run)
        job_name = args.job_name or run_name
        _debug(f"job_name: {job_name}")

        strategy = _resolve_resume_strategy(
            resume_dir=resume_dir,
            overrides=overrides,
            config=config,
        )
        _debug(f"distribution strategy (resume): {strategy}")
        
        print(f"\n{'='*60}")
        print(f"RESUME MODE")
        print(f"{'='*60}")
        print(f"  Resume directory: {resume_dir}")
        print(f"  Run name: {run_name}")
        print(f"  Job name: {job_name}")
        print(f"  Distribution strategy: {strategy}")
        if overrides:
            print(f"  Overrides: {overrides}")
        print(f"{'='*60}\n")
        
        # Update config for resume
        config["job_name"] = job_name
        config["logdir_name"] = run_name
        _debug(f"config[job_name]: {config['job_name']}")
        _debug(f"config[logdir_name]: {config['logdir_name']}")
        
        # Build strategy-aware command for resume_training.py
        resume_args = f"{resume_dir} {' '.join(overrides)}".strip()
        config["python_command"] = _build_training_command(
            strategy=strategy,
            gpus=int(config["gpus"]),
            script_name="resume_training.py",
            script_args=resume_args,
        )
        _debug(f"python_command: {config['python_command']}")
        
        # Set container logdir to the resume directory
        config["container_logdir"] = resume_dir
        _debug(f"container_logdir: {config['container_logdir']}")
        
        # Derive host logdir from container path
        # resume_dir format: /mnt/outputs/run_name or similar
        if resume_dir.startswith(config["container_outputs_base"].rstrip('/')):
            relative_path = resume_dir[len(config["container_outputs_base"].rstrip('/')):]
            config["host_logdir"] = os.path.join(config["host_outputs_dir"], relative_path.lstrip('/'))
            _debug(f"host_logdir (derived): {config['host_logdir']}")
        else:
            # Fallback: use the run_name under host_outputs_dir
            config["host_logdir"] = os.path.join(config["host_outputs_dir"], run_name)
            _debug(f"host_logdir (fallback): {config['host_logdir']}")
        
    else:
        # ======================================================================
        # FRESH TRAINING MODE: Use start_training.py with new run directory
        # ======================================================================
        _debug("Entering FRESH TRAINING MODE branch")
        
        # Load custom config
        _debug(f"Loading config: {args.config_name}")
        cfg = load_config(args.config_name, overrides)
        pprint.pprint(cfg)  # Debug print of final config
        _debug("Config loaded successfully")

        strategy = cfg.get("distribution", {}).get("strategy", "dp")
        if strategy not in {"dp", "ddp"}:
            raise ValueError(f"Unsupported distribution.strategy='{strategy}'. Expected 'dp' or 'ddp'.")
        _debug(f"distribution strategy (fresh): {strategy}")

        # Auto-set legacy multi_gpu only for DP strategy when --gpus > 1
        if int(config["gpus"]) > 1 and strategy == "dp":
            has_multi_gpu_override = any("environment.training.multi_gpu" in o for o in overrides)
            if not has_multi_gpu_override:
                gpu_list = list(range(int(config["gpus"])))
                overrides.append(f"environment.training.multi_gpu=[{','.join(map(str, gpu_list))}]")
                print(
                    f"[AUTO] distribution=dp with --gpus {config['gpus']}, "
                    f"adding environment.training.multi_gpu={gpu_list}"
                )
        
        # Generate run_name using utility
        _debug("Generating run_name...")
        run_name = generate_run_name(cfg, timestamp)
        _debug(f"run_name: {run_name}")
        
        # Set job_name
        job_name = args.job_name or run_name
        _debug(f"job_name: {job_name}")
        
        print(f"\n{'='*60}")
        print(f"FRESH TRAINING MODE")
        print(f"{'='*60}")
        print(f"  Config: {args.config_name}")
        print(f"  Run name: {run_name}")
        print(f"  Job name: {job_name}")
        print(f"  Distribution strategy: {strategy}")
        if overrides:
            print(f"  Overrides: {overrides}")
        print(f"{'='*60}\n")
        
        # Update config
        config["job_name"] = job_name
        config["logdir_name"] = run_name  # For outputs
        config["hydra_config_name"] = args.config_name
        _debug(f"config[job_name]: {config['job_name']}")
        _debug(f"config[logdir_name]: {config['logdir_name']}")
        
        # Add timestamp and run_name to overrides for start_training.py
        all_overrides = overrides + [f"timestamp={timestamp}", f"run_name={run_name}"]
        _debug(f"all_overrides: {all_overrides}")
        
        # Build strategy-aware command for start_training.py
        start_args = f"--config-name {args.config_name} {' '.join(all_overrides)}".strip()
        config["python_command"] = _build_training_command(
            strategy=strategy,
            gpus=int(config["gpus"]),
            script_name="start_training.py",
            script_args=start_args,
        )
        _debug(f"python_command: {config['python_command']}")
        
        # Update config with resolved output_root
        config["container_outputs_dir"] = cfg["environment"]["training"]["output_root"]
        _debug(f"container_outputs_dir: {config['container_outputs_dir']}")
        
        # Derive relative part (strip container's /mnt/)
        if config["container_outputs_dir"].startswith(config["container_prefix"]):
            relative_out = config["container_outputs_dir"][len(config["container_prefix"]):]
            config["host_outputs_dir"] = config["host_base"] + relative_out
            _debug(f"host_outputs_dir: {config['host_outputs_dir']}")
        else:
            raise ValueError(f"output_root '{{config[\"container_outputs_dir\"]}}' does not start with expected container_prefix '{{config[\"container_prefix\"]}}'")

        # Update logdir paths with new logdir_name
        config = update_logdir_paths(config)
        _debug(f"host_logdir: {config.get('host_logdir')}")
        _debug(f"container_logdir: {config.get('container_logdir')}")
    
    # ==========================================================================
    # Common: Submit the job
    # ==========================================================================
    _debug("="*60)
    _debug("Preparing to submit job")
    _debug("="*60)
    _debug(f"Final python_command: {config.get('python_command')}")
    _debug(f"Final host_logdir: {config.get('host_logdir')}")
    _debug(f"Final container_logdir: {config.get('container_logdir')}")
    
    # Initialize runner
    runner = SlurmJobRunner(config)
    _debug("SlurmJobRunner initialized")
    
    # Submit
    _debug(f"Calling submit_job (dry_run={args.dry_run})...")
    runner.submit_job(config, SLURM_TEMPLATE, dry_run=args.dry_run)
    _debug("submit_job complete")

if __name__ == "__main__":
    main()
