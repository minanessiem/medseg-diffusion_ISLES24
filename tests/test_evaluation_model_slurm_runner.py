"""
Tests for the config-driven model evaluation SLURM runner.
"""

import argparse
import unittest

from scripts.evaluation.slurm_runners.run_evaluate_model import (
    build_evaluation_overrides,
    build_python_command,
    build_slurm_config,
)


def _args(**overrides) -> argparse.Namespace:
    defaults = {
        "run_dir": "/mnt/outputs/run with spaces",
        "model_name": "diffusion_chkpt_step_000010",
        "evaluation_config_name": "threshold_sweep_with_oracle",
        "output_dir": None,
        "validation_config": "sliding_window_3d_metrics_subset",
        "val_subset": "val_fast",
        "val_batch_size": 1,
        "show_progress": False,
        "overrides": [],
        "gpus": 1,
        "partition": None,
        "qos": None,
        "cpus_per_task": 32,
        "mem": "96G",
        "time": "06:00:00",
        "dry_run": True,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _base_slurm_config() -> dict:
    return {
        "partition": "test-partition",
        "qos": "test-qos",
        "gpus": 1,
        "time": "01:00:00",
        "cpus_per_task": 8,
        "mem": "16G",
        "host_outputs_dir": "/host/outputs",
        "container_outputs_dir": "/mnt/outputs",
        "logdir_name": "logs",
        "container_image": "/images/default.sqsh",
    }


class TestEvaluateModelSlurmRunner(unittest.TestCase):
    def test_build_evaluation_overrides_uses_validation_preset(self):
        args = _args(output_dir="/mnt/outputs/run/eval", overrides=["evaluation.device=cuda"])

        overrides = build_evaluation_overrides(args)

        self.assertEqual(overrides[0], "evaluation.run_dir=/mnt/outputs/run with spaces")
        self.assertIn("validation=sliding_window_3d_metrics_subset", overrides)
        self.assertIn("validation.val_batch_size=1", overrides)
        self.assertIn("dataset.active_subsets.val=val_fast", overrides)
        self.assertEqual(overrides[-1], "evaluation.device=cuda")

    def test_build_python_command_quotes_paths_and_overrides(self):
        args = _args(
            output_dir="/mnt/outputs/run/eval dir",
            overrides=["evaluation.threshold_protocol.thresholds=0.05:0.90:0.05"],
        )

        command = build_python_command(args)

        self.assertTrue(command.startswith("python3 -m scripts.evaluation.evaluate_model"))
        self.assertIn("--evaluation-config-name threshold_sweep_with_oracle", command)
        self.assertIn("'evaluation.run_dir=/mnt/outputs/run with spaces'", command)
        self.assertIn("'evaluation.output_dir=/mnt/outputs/run/eval dir'", command)
        self.assertIn("evaluation.threshold_protocol.thresholds=0.05:0.90:0.05", command)

    def test_build_slurm_config_places_logs_under_run_analysis(self):
        args = _args(
            run_dir="/mnt/outputs/experiment/run_001",
            model_name="model/name with spaces",
            partition="custom",
            qos="urgent",
            cpus_per_task=64,
            mem="128G",
            time="12:00:00",
        )

        config = build_slurm_config(args, base_config=_base_slurm_config())

        self.assertEqual(config["partition"], "custom")
        self.assertEqual(config["qos"], "urgent")
        self.assertEqual(config["cpus_per_task"], 64)
        self.assertEqual(config["mem"], "128G")
        self.assertEqual(config["time"], "12:00:00")
        self.assertTrue(str(config["job_name"]).startswith("eval_model_model_name_with_spaces"))
        self.assertIn("experiment/run_001/analysis/slurm_logs/evaluate_model_", str(config["logdir_name"]))
        self.assertTrue(str(config["host_logdir"]).startswith("/host/outputs/experiment/run_001/analysis/slurm_logs"))

    def test_build_slurm_config_applies_generic_base_config_overrides(self):
        args = _args(container_image="/images/eval.sqsh")

        config = build_slurm_config(args, base_config=_base_slurm_config())

        self.assertEqual(config["container_image"], "/images/eval.sqsh")


if __name__ == "__main__":
    unittest.main(verbosity=2)
