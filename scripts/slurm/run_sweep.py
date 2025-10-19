#!/usr/bin/env python3

import os
import argparse
from typing import List, Optional
import itertools
from datetime import datetime

from scripts.slurm.base_run_config import BASE_CONFIG, SLURM_TEMPLATE, update_logdir_paths
from scripts.slurm.job_runner import SlurmJobRunner
from scripts.slurm.utils.commandline_utils import add_config_arguments, update_config_from_args

def generate_n_modality_combinations(
    base_modalities: List[str],
    optional_modalities: List[str],
    n_choose: int
) -> List[List[str]]:
    """
    Generate combinations of base_modalities + exactly n_choose from optional_modalities.
    
    Args:
        base_modalities: List of modalities to include in all combinations
        optional_modalities: Pool of additional modalities to choose from
        n_choose: Exact number of optional modalities to include
    
    Returns:
        List of modality combinations, where each combination is a sorted list
    """
    if n_choose > len(optional_modalities):
        raise ValueError(f"Cannot choose {n_choose} modalities from {len(optional_modalities)}")
    
    optional_combinations = list(itertools.combinations(optional_modalities, n_choose))
    all_combinations = []
    
    for opt_combo in optional_combinations:
        combination = base_modalities + list(opt_combo)
        combination.sort()
        all_combinations.append(combination)
    
    return all_combinations

def generate_powerset_combinations(
    base_modalities: List[str],
    optional_modalities: List[str],
    max_size: Optional[int] = None
) -> List[List[str]]:
    """
    Generate all possible combinations of base_modalities + optional_modalities up to max_size.
    
    Args:
        base_modalities: List of modalities to include in all combinations
        optional_modalities: Pool of additional modalities to choose from
        max_size: Maximum number of optional modalities to include. If None, includes all sizes
    
    Returns:
        List of modality combinations, where each combination is a sorted list
    """
    if max_size is None:
        max_size = len(optional_modalities)
    elif max_size > len(optional_modalities):
        max_size = len(optional_modalities)
    
    all_combinations = []
    for n in range(1, max_size + 1):
        combinations = generate_n_modality_combinations(
            base_modalities, optional_modalities, n
        )
        all_combinations.extend(combinations)
    
    return all_combinations

def create_runstring(modalities: List[str]) -> str:
    """Create a readable runstring from configuration."""
    return "_".join(sorted(modalities))

def create_slurm_job_name(config: dict, modalities_str: str) -> str:
    """Creates a job name for SLURM based on key configuration."""
    model_name = config["model_name"]
    fold = config["fold"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_name}_{modalities_str}_{fold}_{timestamp}"

def main():
    parser = argparse.ArgumentParser(description='Submit SLURM jobs for model training')
    
    # Add sweep-specific arguments
    parser.add_argument('--dry-run', action='store_true', 
                       help='Print jobs that would be submitted without actually submitting them')
    parser.add_argument('--powerset', action='store_true',
                       help='Generate all possible combinations up to max-choose')
    parser.add_argument('--max-choose', type=int, default=1,
                       help='Maximum number of optional modalities to include')
    parser.add_argument('--base-modalities', type=str, default=None,
                       help='Comma-separated list of base modalities (e.g., "NCCT_wg,CBF_3,CBF_mask_3")')
    parser.add_argument('--optional-modalities', type=str, default=None,
                       help='Comma-separated list of optional modalities (e.g., "TMAX_4,TMAX_5,TMAX_6")')
    
    # Add configuration override arguments
    add_config_arguments(parser, BASE_CONFIG)
    
    args = parser.parse_args()
    
    # Update BASE_CONFIG with command line arguments
    config = update_config_from_args(BASE_CONFIG.copy(), args, update_logdir_paths)
    
    # Validate arguments
    if args.base_modalities is None and args.optional_modalities is None:
        parser.error("At least one of --base-modalities or --optional-modalities must be specified")
    
    # Parse modality lists
    base_modalities = [m.strip() for m in args.base_modalities.split(',')] if args.base_modalities else []
    optional_modalities = [m.strip() for m in args.optional_modalities.split(',')] if args.optional_modalities else []
    
    # Handle single run case (only base modalities, no optional modalities)
    if not optional_modalities:
        modality_combinations = [base_modalities] if base_modalities else []
    # Handle cases with optional modalities (with or without base modalities)
    else:
        if args.powerset:
            modality_combinations = generate_powerset_combinations(
                base_modalities=base_modalities,
                optional_modalities=optional_modalities,
                max_size=args.max_choose
            )
        else:
            modality_combinations = generate_n_modality_combinations(
                base_modalities=base_modalities,
                optional_modalities=optional_modalities,
                n_choose=args.max_choose
            )
    
    # Initialize SLURM job runner
    runner = SlurmJobRunner(config)
    
    # Create and verify output directory
    outputs_dir = config["host_logdir"]
    os.makedirs(outputs_dir, exist_ok=True)
    print(f"\nOutput directory:")
    print(f"  Path: {outputs_dir}")
    print(f"  Absolute path: {os.path.abspath(outputs_dir)}")
    print(f"  Exists: {os.path.exists(outputs_dir)}")
    print(f"  Is directory: {os.path.isdir(outputs_dir)}")
    print(f"  Current working directory: {os.getcwd()}")
    
    # Get existing runstrings
    existing_runstrings = runner.get_existing_runstrings()
    
    # Print combinations with their sizes and existing status
    print("\nGenerated combinations:")
    for combo in modality_combinations:
        n_optional = len(combo) - len(base_modalities) if base_modalities else len(combo)
        runstring = create_runstring(combo)
        status = " (EXISTS - would skip)" if runstring in existing_runstrings else ""
        print(f"  {combo} (+ {n_optional} optional){status}")
    
    print(f"\nTotal number of combinations: {len(modality_combinations)}")
    print(f"Number of existing combinations: {sum(1 for combo in modality_combinations if create_runstring(combo) in existing_runstrings)}")
    print(f"Number of new combinations: {sum(1 for combo in modality_combinations if create_runstring(combo) not in existing_runstrings)}")
    
    if args.dry_run:
        print("\nThis is a dry run - no jobs will be submitted")
        return
    
    # Submit jobs
    for modalities in modality_combinations:
        runstring = create_runstring(modalities)
        
        if runstring in existing_runstrings:
            print(f"Skipping existing runstring: {runstring}")
            continue
            
        job_name = create_slurm_job_name(config, runstring)
        submitted_config = {
            **config,
            "job_name": job_name,
            "modalities": ",".join(modalities),
            "in_channels": len(modalities)
        }
        
        runner.submit_job(submitted_config, SLURM_TEMPLATE, dry_run=args.dry_run)

if __name__ == "__main__":
    main()