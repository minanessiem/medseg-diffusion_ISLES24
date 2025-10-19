import argparse
import os
from typing import Dict

def str2bool(v: str) -> bool:
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def add_config_arguments(parser: argparse.ArgumentParser, base_config: Dict) -> None:
    """Add all base_config parameters as optional arguments."""
    # Add a config group for better help message organization
    config_group = parser.add_argument_group('configuration overrides')
    
    # Add an argument for each base_config parameter
    for key, value in base_config.items():
        arg_name = f"--{key.replace('_', '-')}"
        
        # Determine the type from the default value
        if isinstance(value, bool):
            config_group.add_argument(arg_name, type=str2bool, default=None,
                                    help=f'Override {key} (default: {value})')
        elif isinstance(value, (int, float, str)):
            config_group.add_argument(arg_name, type=type(value), default=None,
                                    help=f'Override {key} (default: {value})')

def update_config_from_args(config: Dict, args: argparse.Namespace, update_dict_fn: callable = None) -> Dict:
    """Update configuration with command line arguments."""
    # Convert args to dict, excluding None values
    arg_dict = {k: v for k, v in vars(args).items() if v is not None}
    
    # Update config with non-None argument values
    for key, value in arg_dict.items():
        # Convert hyphenated args back to underscores
        config_key = key.replace('-', '_')
        if config_key in config:
            print(f"Overriding {config_key}: {config[config_key]} -> {value}")
            config[config_key] = value
    
    # Update dictionary if an update function was provided
    if update_dict_fn is not None:
        old_config = config.copy()
        config = update_dict_fn(config)
        # Print any values that changed during the update
        for key, new_value in config.items():
            if key in old_config and old_config[key] != new_value:
                print(f"Updated {key}: {old_config[key]} -> {new_value}")
            elif key not in old_config:
                print(f"Added {key}: {new_value}")
    
    return config