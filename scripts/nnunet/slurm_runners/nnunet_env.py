"""
Shared nnU-Net environment configuration for SLURM runners.

This module defines:
- nnU-Net environment variables (paths on the cluster)
- Modified SLURM template that exports these before running commands
- Default resource configurations per command type
"""

import os

# ============================================================================
# nnU-Net Environment Variables
# ============================================================================
# These paths are where nnU-Net expects to find/store data inside the container

NNUNET_ENV = {
    "nnUNet_raw": "/mnt/datasets/nnunet_raw",
    "nnUNet_preprocessed": "/mnt/datasets/nnunet_preprocessed",
    "nnUNet_results": "/mnt/outputs/nnunet_results",
    "nnUNet_compile": "False",  # Disable torch.compile for compatibility
}


def get_env_export_string() -> str:
    """Generate bash export statements for nnU-Net environment variables."""
    exports = [f"export {key}={value}" for key, value in NNUNET_ENV.items()]
    return " && ".join(exports)


# ============================================================================
# Default Resource Configurations
# ============================================================================
# Sensible defaults for each nnU-Net command type

COMMAND_DEFAULTS = {
    "preprocess": {
        "time": "02:00:00",
        "cpus_per_task": 64,
        "mem": "128G",
        "description": "CPU/memory intensive, no GPU needed for preprocessing",
    },
    "train": {
        "time": "47:00:00",
        "cpus_per_task": 128,
        "mem": "256G",
        "description": "Long-running GPU training",
    },
    "predict": {
        "time": "04:00:00",
        "cpus_per_task": 32,
        "mem": "64G",
        "description": "GPU inference, faster than training",
    },
}


# ============================================================================
# Modified SLURM Template for nnU-Net
# ============================================================================
# This template includes nnU-Net environment variable exports

# Import base config for reference
from scripts.slurm.base_run_config import BASE_CONFIG

HOME_DIR = os.path.expanduser("~")
ssd_store_path = os.environ.get("SSD_STORE")

NNUNET_SLURM_TEMPLATE = """#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --gres=gpu:{gpus}
#SBATCH --time={time}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --output={output_file}
#SBATCH --error={error_file}

# Define paths
CODE_DIR={host_code_dir}
DATASETS_DIR={host_datasets_dir}
OUTPUTS_DIR={host_outputs_dir}
IMAGE='{container_image}'

# Ensure log directories exist
mkdir -p logs

# Set ulimit
ulimit -n 4096

# Execute the job with nnU-Net environment variables
srun \\
  --container-mounts=$CODE_DIR:{container_code_dir},$DATASETS_DIR:{container_datasets_dir},$OUTPUTS_DIR:{container_outputs_dir} \\
  --container-image=$IMAGE \\
  bash -c "cd {container_code_dir}/medseg-diffusion_ISLES24 && export PYTHONNOUSERSITE=1 && export TORCH_MULTIPROCESSING_SHARING_STRATEGY=file_system && {nnunet_env_exports} && {command}"
"""

