#!/usr/bin/env python3
"""
Submit config-driven repository-model evaluation as a SLURM job.

This runner intentionally avoids Hydra/OmegaConf on the submission side. It
builds a plain command string for `python3 -m scripts.evaluation.evaluate_model`
and forwards evaluation overrides as `key=value` arguments to the container job.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from datetime import datetime
from typing import Dict, List, Optional


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scripts.slurm.utils.commandline_utils import add_config_arguments, update_config_from_args


def parse_arguments() -> argparse.Namespace:
    from scripts.slurm.base_run_config import BASE_CONFIG

    parser = argparse.ArgumentParser(
        description="Submit config-driven repository-model evaluation as a SLURM job.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--run-dir", required=True, help="Run directory with .hydra/config.yaml")
    parser.add_argument("--model-name", required=True, help="Checkpoint name, with or without .pth")
    parser.add_argument(
        "--evaluation-config-name",
        type=str,
        default="threshold_sweep_with_oracle",
        help="Config preset under configs/evaluation/ (default: threshold_sweep_with_oracle)",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Optional evaluation output directory")
    parser.add_argument(
        "--validation-config",
        type=str,
        default="sliding_window_3d_metrics_subset",
        help="Validation config group override (default: sliding_window_3d_metrics_subset)",
    )
    parser.add_argument(
        "--val-subset",
        type=str,
        default=None,
        help="Optional dataset.active_subsets.val override, e.g. val_fast or val_full",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=1,
        help="validation.val_batch_size override (default: 1 for variable-shape 3D volumes)",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Enable tqdm progress inside the evaluation job.",
    )
    parser.add_argument(
        "--override",
        dest="overrides",
        nargs="*",
        default=[],
        help=(
            "Additional key=value overrides forwarded after convenience overrides. "
            "Example: --override evaluation.threshold_protocol.fixed_threshold=0.4"
        ),
    )

    parser.add_argument("--gpus", type=int, default=None, help="Number of GPUs")
    parser.add_argument("--partition", type=str, default=None, help="SLURM partition override")
    parser.add_argument("--qos", type=str, default=None, help="SLURM QoS override")
    parser.add_argument("--cpus-per-task", type=int, default=None, help="CPUs per task")
    parser.add_argument("--mem", type=str, default=None, help="Memory allocation")
    parser.add_argument("--time", type=str, default=None, help="Time limit")
    parser.add_argument("--dry-run", action="store_true", help="Print configuration without submitting")

    excluded_params = {"gpus", "partition", "qos", "cpus_per_task", "mem", "time"}
    filtered_config = {key: value for key, value in BASE_CONFIG.items() if key not in excluded_params}
    add_config_arguments(parser, filtered_config)

    return parser.parse_args()


def build_evaluation_overrides(args: argparse.Namespace) -> List[str]:
    """Build ordered evaluate_model key=value overrides."""
    overrides = [
        f"evaluation.run_dir={args.run_dir}",
        f"evaluation.model_name={args.model_name}",
        f"validation={args.validation_config}",
        f"validation.val_batch_size={int(args.val_batch_size)}",
        f"evaluation.show_progress={str(bool(args.show_progress)).lower()}",
    ]
    if args.output_dir:
        overrides.append(f"evaluation.output_dir={args.output_dir}")
    if args.val_subset:
        overrides.append(f"dataset.active_subsets.val={args.val_subset}")
    overrides.extend(str(override) for override in args.overrides)
    return overrides


def build_python_command(args: argparse.Namespace) -> str:
    """Build the command executed inside the SLURM container."""
    cmd_parts = [
        "python3 -m scripts.evaluation.evaluate_model",
        "--evaluation-config-name",
        shlex.quote(str(args.evaluation_config_name)),
    ]
    cmd_parts.extend(shlex.quote(override) for override in build_evaluation_overrides(args))
    return " ".join(cmd_parts)


def build_slurm_config(args: argparse.Namespace, base_config: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Build the SLURM config dictionary for submission or dry-run output."""
    if base_config is None:
        from scripts.slurm.base_run_config import BASE_CONFIG, update_logdir_paths

        config = update_config_from_args(BASE_CONFIG.copy(), args, update_logdir_paths)
    else:
        update_logdir_paths = _update_logdir_paths
        config = update_config_from_args(dict(base_config), args, update_logdir_paths)

    config["python_command"] = build_python_command(args)
    if args.gpus is not None:
        config["gpus"] = int(args.gpus)
    if args.cpus_per_task is not None:
        config["cpus_per_task"] = int(args.cpus_per_task)
    if args.mem is not None:
        config["mem"] = str(args.mem)
    if args.time is not None:
        config["time"] = str(args.time)
    if args.partition:
        config["partition"] = args.partition
    if args.qos:
        config["qos"] = args.qos

    model_name_short = _sanitize_job_token(args.model_name, max_len=26)
    config["job_name"] = f"eval_model_{model_name_short}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["logdir_name"] = os.path.join(
        _relative_run_log_root(args.run_dir, config),
        "analysis",
        "slurm_logs",
        f"evaluate_model_{timestamp}",
    )

    return update_logdir_paths(config)


def main() -> None:
    args = parse_arguments()
    config = build_slurm_config(args)
    python_command = str(config["python_command"])

    print("\n" + "=" * 70)
    print("CONFIG-DRIVEN MODEL EVALUATION SLURM JOB")
    print("=" * 70)
    print(f"  Run dir:      {args.run_dir}")
    print(f"  Model:        {args.model_name}")
    print(f"  Eval config:  {args.evaluation_config_name}")
    print(f"  Validation:   {args.validation_config}")
    print(f"  Val batch:    {args.val_batch_size}")
    if args.val_subset:
        print(f"  Val subset:   {args.val_subset}")
    if args.output_dir:
        print(f"  Output dir:   {args.output_dir}")
    if args.overrides:
        print(f"  Overrides:    {args.overrides}")
    print("-" * 70)
    print("  Resources:")
    print(f"    GPUs:       {config['gpus']}")
    print(f"    CPUs:       {config['cpus_per_task']}")
    print(f"    Memory:     {config['mem']}")
    print(f"    Time:       {config['time']}")
    print(f"    Partition:  {config['partition']}")
    print("-" * 70)
    print("  Python command:")
    print(f"    {python_command}")
    print("=" * 70)

    if args.dry_run:
        config["output_file"] = f"{config['host_logdir']}/output.out"
        config["error_file"] = f"{config['host_logdir']}/error.err"
        print("\n[DRY RUN] Configuration:")
        for key in [
            "job_name",
            "partition",
            "qos",
            "gpus",
            "time",
            "cpus_per_task",
            "mem",
            "host_logdir",
            "python_command",
        ]:
            print(f"  {key}: {config[key]}")
        print("\n[DRY RUN] No job submitted.")
        return

    from scripts.slurm.base_run_config import SLURM_TEMPLATE
    from scripts.slurm.job_runner import SlurmJobRunner

    runner = SlurmJobRunner(config)
    job_id = runner.submit_job(config, SLURM_TEMPLATE, dry_run=False)
    if job_id:
        print(f"\nJob submitted successfully: {job_id}")
        print(f"  Monitor with: squeue -j {job_id}")
        print(f"  Logs at: {config['host_logdir']}")
    else:
        print("\nJob submission failed")
        sys.exit(1)


def _relative_run_log_root(run_dir: str, config: Dict[str, object]) -> str:
    container_outputs = str(config.get("container_outputs_dir", "/mnt/outputs")).rstrip("/")
    run_dir_clean = str(run_dir).rstrip("/")
    if run_dir_clean.startswith(container_outputs):
        relative = run_dir_clean[len(container_outputs) :].lstrip("/")
        if relative:
            return relative
    return os.path.basename(run_dir_clean)


def _update_logdir_paths(config: Dict[str, object]) -> Dict[str, object]:
    config["container_logdir"] = os.path.join(str(config["container_outputs_dir"]), str(config["logdir_name"]))
    config["host_logdir"] = os.path.join(str(config["host_outputs_dir"]), str(config["logdir_name"]))
    return config


def _sanitize_job_token(value: str, max_len: int) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(value))
    safe = safe.strip("_") or "model"
    return safe[:max_len]


if __name__ == "__main__":
    main()
