"""
Tests for config-driven repository-model evaluation pipeline.
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from scripts.evaluation.core.contracts import SliceSample, VolumeSample
from scripts.evaluation.core.evaluation_pipeline import (
    _resolve_slice_metric_names,
    _resolve_volume_metric_names,
    build_model_evaluation_request,
    run_model_evaluation,
)


class DummyModel(nn.Module):
    def forward(self, x):
        return x


def _make_cfg(tmp: str, mode: str = "fixed"):
    run_dir = Path(tmp) / "run"
    output_dir = Path(tmp) / "eval"
    checkpoint_dir = run_dir / "models" / "best"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "best_model.pth").write_bytes(b"checkpoint")
    return OmegaConf.create(
        {
            "evaluation": {
                "input_source": "live_model",
                "run_dir": str(run_dir),
                "model_name": "best_model",
                "output_dir": str(output_dir),
                "device": "cpu",
                "levels": ["volume"],
                "checkpoint": {"use_ema": False},
                "threshold_protocol": {
                    "mode": mode,
                    "thresholds": "0.3,0.5",
                    "fixed_threshold": 0.5,
                    "primary": {
                        "level": "volume",
                        "metric": "DiceNativeCoefficient",
                        "statistic": "mean",
                        "direction": "max",
                    },
                },
                "metrics_3d": {
                    "names": [
                        "DiceNativeCoefficient",
                        "PredictedVolumeMm3",
                        "GroundTruthVolumeMm3",
                    ]
                },
                "reporting": {"write_config": True},
                "show_progress": False,
            },
            "data_mode": {
                "dim": "3d",
                "loader_mode": "full_volumes_3d",
            },
            "diffusion": {"type": "Discriminative"},
            "dataset": {"active_subsets": {"val": "val_fast"}},
            "validation": {"inference": {"mode": "direct"}},
        }
    )


def _volume_sample(case_id: str = "case_a"):
    return VolumeSample(
        case_id=case_id,
        volume_id=case_id,
        prediction_volume=torch.ones(1, 2, 2, 2),
        ground_truth_volume=torch.ones(1, 2, 2, 2),
        metadata={"source_spacing_xyz": (1.0, 1.0, 1.0)},
    )


def _slice_sample(slice_id: str = "case_a_slice0", slice_index: int = 0):
    return SliceSample(
        case_id="case_a",
        slice_id=slice_id,
        volume_id="case_a",
        slice_index=slice_index,
        prediction_prob=torch.ones(1, 2, 2),
        ground_truth_mask=torch.ones(1, 2, 2),
        metadata={"source": "test_slice"},
    )


def _mock_metric_values(pred, gt, threshold, metric_configs=None, metric_names=None):
    del pred, gt, metric_configs
    dice_by_threshold = {0.3: 0.4, 0.5: 0.8}
    values = {
        "DiceNativeCoefficient": dice_by_threshold[round(float(threshold), 1)],
        "SurfaceDiceMonai": 0.25,
        "HausdorffDistance95Native": 12.0,
        "PredictedVolumeMm3": 8.0,
        "GroundTruthVolumeMm3": 10.0,
    }
    if metric_names is None:
        return values
    return {name: values[name] for name in metric_names}


def _mock_slice_metric_values(pred, gt, threshold, metric_names=None):
    del pred, gt
    dice_by_threshold = {0.3: 0.2, 0.5: 0.9}
    values = {
        "Dice2DForegroundOnly": dice_by_threshold[round(float(threshold), 1)],
        "VoxelF1Score2D": 0.7,
    }
    if metric_names is None:
        return values
    return {name: values[name] for name in metric_names}


def _make_slice_metric_side_effect(values_by_slice):
    call_index = {"value": 0}
    slice_order = list(values_by_slice)

    def _side_effect(pred, gt, threshold, metric_names=None):
        del pred, gt
        threshold_key = round(float(threshold), 1)
        sample_index = call_index["value"] // 2
        call_index["value"] += 1
        slice_id = slice_order[sample_index]
        values = {
            "Dice2DForegroundOnly": values_by_slice[slice_id][threshold_key],
            "VoxelF1Score2D": 0.5,
        }
        if metric_names is None:
            return values
        return {name: values[name] for name in metric_names}

    return _side_effect


class TestEvaluationPipeline(unittest.TestCase):
    def test_build_request_validates_required_fields(self):
        cfg = OmegaConf.create(
            {
                "evaluation": {"model_name": "best_model"},
                "data_mode": {"dim": "3d"},
                "diffusion": {"type": "Discriminative"},
            }
        )

        with self.assertRaises(ValueError):
            build_model_evaluation_request(cfg)

    def test_build_request_resolves_checkpoint_and_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="sweep")
            request = build_model_evaluation_request(cfg)

        self.assertEqual(request.model_name, "best_model")
        self.assertEqual(request.device, "cpu")
        self.assertEqual(request.data_dim, "3d")
        self.assertEqual(request.diffusion_type, "Discriminative")
        self.assertEqual(request.threshold_protocol.mode, "sweep")
        self.assertTrue(str(request.checkpoint_path).endswith("best_model.pth"))

    def test_build_request_normalizes_level_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="sweep")
            cfg.evaluation.levels = ["volumes"]
            cfg.evaluation.threshold_protocol.primary.level = "volumes"

            request = build_model_evaluation_request(cfg)

        self.assertEqual(list(request.levels), ["volume"])
        self.assertEqual(request.threshold_protocol.primary.level, "volume")

    def test_build_request_accepts_2d_slice_level_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="sweep")
            cfg.data_mode.dim = "2d"
            cfg.data_mode.loader_mode = "online_slices_3d_to_2d"
            cfg.evaluation.levels = ["slices"]
            cfg.evaluation.threshold_protocol.primary.level = "slices"
            cfg.evaluation.threshold_protocol.primary.metric = "Dice2DForegroundOnly"

            request = build_model_evaluation_request(cfg)

        self.assertEqual(request.data_dim, "2d")
        self.assertEqual(list(request.levels), ["slice"])
        self.assertEqual(request.threshold_protocol.primary.level, "slice")

    def test_build_request_rejects_primary_level_not_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="sweep")
            cfg.evaluation.levels = ["slice"]
            cfg.evaluation.threshold_protocol.primary.level = "volume"

            with self.assertRaises(ValueError) as ctx:
                build_model_evaluation_request(cfg)

        self.assertIn("primary.level must be included", str(ctx.exception))

    def test_build_request_rejects_2d_volume_level_until_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="sweep")
            cfg.data_mode.dim = "2d"
            cfg.data_mode.loader_mode = "online_slices_3d_to_2d"
            cfg.evaluation.levels = ["volume"]
            cfg.evaluation.threshold_protocol.primary.level = "volume"

            with self.assertRaises(ValueError) as ctx:
                build_model_evaluation_request(cfg)

        self.assertIn("2D live-model evaluation currently supports slice-level", str(ctx.exception))

    def test_build_request_rejects_2d_mixed_levels_until_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="sweep")
            cfg.data_mode.dim = "2d"
            cfg.data_mode.loader_mode = "online_slices_3d_to_2d"
            cfg.evaluation.levels = ["slice", "volume"]
            cfg.evaluation.threshold_protocol.primary.level = "slice"

            with self.assertRaises(ValueError) as ctx:
                build_model_evaluation_request(cfg)

        self.assertIn("2D live-model evaluation currently supports slice-level", str(ctx.exception))

    def test_build_request_rejects_3d_slice_level_until_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="sweep")
            cfg.evaluation.levels = ["slice"]
            cfg.evaluation.threshold_protocol.primary.level = "slice"

            with self.assertRaises(ValueError) as ctx:
                build_model_evaluation_request(cfg)

        self.assertIn("3D live-model evaluation currently supports volume-level", str(ctx.exception))

    def test_unsupported_3d_diffusion_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="fixed")
            cfg.diffusion.type = "OpenAI_DDPM"
            with self.assertRaises(ValueError) as ctx:
                build_model_evaluation_request(cfg)

        self.assertIn("3D live-model evaluation", str(ctx.exception))

    def test_validation_metric_subset_resolves_to_class_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="fixed")
            del cfg.evaluation.metrics_3d
            cfg.validation.metrics = [
                {
                    "name": "ThreeDMetricsAggregator",
                    "params": {
                        "enabled_metrics": [
                            "dice_3d",
                            "surface_dice_monai_3d",
                            "hd95_3d",
                        ]
                    },
                }
            ]

            metric_names = _resolve_volume_metric_names(cfg)

        self.assertEqual(
            metric_names,
            (
                "DiceNativeCoefficient",
                "SurfaceDiceMonai",
                "HausdorffDistance95Native",
            ),
        )

    def test_evaluation_metric_override_rejects_validation_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="fixed")
            cfg.evaluation.metrics_3d.names = ["dice_3d"]

            with self.assertRaises(ValueError) as ctx:
                _resolve_volume_metric_names(cfg)

        self.assertIn("must use 3D metric class names", str(ctx.exception))

    def test_slice_metric_override_accepts_class_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="fixed")
            cfg.evaluation.metrics_2d = {
                "names": ["Dice2DForegroundOnly", "VoxelF1Score2D"]
            }

            metric_names = _resolve_slice_metric_names(cfg)

        self.assertEqual(metric_names, ("Dice2DForegroundOnly", "VoxelF1Score2D"))

    def test_slice_metric_override_rejects_validation_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="fixed")
            cfg.evaluation.metrics_2d = {"names": ["dice_2d_fg"]}

            with self.assertRaises(ValueError) as ctx:
                _resolve_slice_metric_names(cfg)

        self.assertIn("must use 2D metric class names", str(ctx.exception))

    def test_slice_metrics_resolve_from_validation_progress_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="fixed")
            cfg.validation.metrics = [{"name": "SliceWiseMetricsAggregator", "params": {}}]
            cfg.validation.progress_metrics = ["dice_2d_fg", "f1_2d"]

            metric_names = _resolve_slice_metric_names(cfg)

        self.assertEqual(metric_names, ("Dice2DForegroundOnly", "VoxelF1Score2D"))

    def test_slice_metrics_resolve_from_validation_enabled_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="fixed")
            cfg.validation.metrics = [
                {
                    "name": "SliceWiseMetricsAggregator",
                    "params": {"enabled_metrics": ["dice_2d_fg", "precision_2d"]},
                }
            ]

            metric_names = _resolve_slice_metric_names(cfg)

        self.assertEqual(metric_names, ("Dice2DForegroundOnly", "VoxelPrecision2D"))

    def test_fixed_protocol_writes_expected_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="fixed")
            result = self._run_mocked_pipeline(cfg)
            output_dir = Path(result["output_dir"])

            self.assertTrue((output_dir / "canonical_results.json").exists())
            self.assertTrue((output_dir / "evaluation_summary.txt").exists())
            self.assertTrue((output_dir / "resolved_evaluation_config.yaml").exists())
            self.assertTrue((output_dir / "volume_metrics_per_threshold.csv").exists())
            self.assertTrue((output_dir / "per_case_threshold_metrics.csv").exists())
            self.assertIsNone(result["oracle_csv_path"])

            payload = json.loads((output_dir / "canonical_results.json").read_text())

        self.assertEqual(payload["protocol"]["mode"], "fixed")
        self.assertEqual(payload["protocol"]["thresholds_evaluated"], [0.5])
        self.assertIsNone(payload["threshold_analysis"]["best_global_threshold"])
        self.assertEqual(payload["data_summary"]["total_volumes"], 1)

    def test_pipeline_uses_validation_metric_subset_with_class_name_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="fixed")
            del cfg.evaluation.metrics_3d
            cfg.validation.metrics = [
                {
                    "name": "ThreeDMetricsAggregator",
                    "params": {
                        "enabled_metrics": [
                            "dice_3d",
                            "surface_dice_monai_3d",
                            "hd95_3d",
                        ]
                    },
                }
            ]
            result = self._run_mocked_pipeline(cfg)
            payload = json.loads(Path(result["json_path"]).read_text())

        metric_names = payload["metrics"]["volume_level"]["metric_names"]
        self.assertEqual(
            metric_names,
            [
                "DiceNativeCoefficient",
                "SurfaceDiceMonai",
                "HausdorffDistance95Native",
            ],
        )
        threshold_metrics = payload["metrics"]["volume_level"]["threshold_results"][0]["metrics"]
        self.assertIn("DiceNativeCoefficient", threshold_metrics)
        self.assertIn("SurfaceDiceMonai", threshold_metrics)
        self.assertIn("HausdorffDistance95Native", threshold_metrics)
        self.assertNotIn("dice_3d", threshold_metrics)

    def test_sweep_protocol_selects_global_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="sweep")
            result = self._run_mocked_pipeline(cfg)
            payload = json.loads(Path(result["json_path"]).read_text())

        best = payload["threshold_analysis"]["best_global_threshold"]
        self.assertEqual(best["threshold"], 0.5)
        self.assertAlmostEqual(best["selected_statistic_value"], 0.8)
        self.assertEqual(result["selected_global_threshold"], 0.5)

    def test_sweep_with_oracle_writes_oracle_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _make_cfg(tmp, mode="sweep_with_oracle")
            result = self._run_mocked_pipeline(cfg)
            oracle_path = Path(result["oracle_csv_path"])
            self.assertTrue(oracle_path.exists())
            payload = json.loads(Path(result["json_path"]).read_text())
            with oracle_path.open("r", encoding="utf-8", newline="") as handle:
                oracle_rows = list(csv.DictReader(handle))

        self.assertEqual(len(oracle_rows), 1)
        self.assertEqual(oracle_rows[0]["case_id"], "case_a")
        self.assertEqual(float(oracle_rows[0]["threshold"]), 0.5)
        self.assertEqual(payload["threshold_analysis"]["oracle_per_case"]["case_count"], 1)

    def test_2d_fixed_protocol_writes_slice_level_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._make_2d_cfg(tmp, mode="fixed")
            result = self._run_mocked_2d_pipeline(cfg)
            output_dir = Path(result["output_dir"])

            self.assertTrue((output_dir / "canonical_results.json").exists())
            self.assertTrue((output_dir / "slice_metrics_per_threshold.csv").exists())
            self.assertTrue((output_dir / "per_case_threshold_metrics.csv").exists())
            self.assertIsNone(result["volume_csv_path"])
            payload = json.loads(Path(result["json_path"]).read_text())

        self.assertEqual(payload["data_summary"]["total_slices"], 1)
        self.assertIn("slice_level", payload["metrics"])
        self.assertNotIn("volume_level", payload["metrics"])
        self.assertEqual(
            payload["metrics"]["slice_level"]["metric_names"],
            ["Dice2DForegroundOnly", "VoxelF1Score2D"],
        )

    def test_2d_sweep_with_oracle_uses_slice_primary_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._make_2d_cfg(tmp, mode="sweep_with_oracle")
            result = self._run_mocked_2d_pipeline(cfg)
            payload = json.loads(Path(result["json_path"]).read_text())
            with Path(result["oracle_csv_path"]).open("r", encoding="utf-8", newline="") as handle:
                oracle_rows = list(csv.DictReader(handle))

        best = payload["threshold_analysis"]["best_global_threshold"]
        self.assertEqual(best["threshold"], 0.5)
        self.assertEqual(result["selected_global_threshold"], 0.5)
        self.assertEqual(len(oracle_rows), 1)
        self.assertEqual(oracle_rows[0]["level"], "slice")
        self.assertEqual(oracle_rows[0]["case_id"], "case_a_slice0")
        self.assertEqual(oracle_rows[0]["selected_metric"], "Dice2DForegroundOnly")
        self.assertEqual(float(oracle_rows[0]["threshold"]), 0.5)

    def test_2d_oracle_selects_one_threshold_per_slice_record(self):
        samples = [
            ("single_case", _slice_sample("case_a_slice0", slice_index=0)),
            ("single_case", _slice_sample("case_a_slice1", slice_index=1)),
        ]
        side_effect = _make_slice_metric_side_effect(
            {
                "case_a_slice0": {0.3: 0.9, 0.5: 0.1},
                "case_a_slice1": {0.3: 0.2, 0.5: 0.8},
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._make_2d_cfg(tmp, mode="sweep_with_oracle")
            result = self._run_mocked_2d_pipeline(
                cfg,
                samples=samples,
                metric_side_effect=side_effect,
            )
            payload = json.loads(Path(result["json_path"]).read_text())
            with Path(result["oracle_csv_path"]).open("r", encoding="utf-8", newline="") as handle:
                oracle_rows = list(csv.DictReader(handle))

        by_slice = {row["case_id"]: row for row in oracle_rows}
        self.assertEqual(set(by_slice), {"case_a_slice0", "case_a_slice1"})
        self.assertEqual(float(by_slice["case_a_slice0"]["threshold"]), 0.3)
        self.assertEqual(float(by_slice["case_a_slice1"]["threshold"]), 0.5)
        self.assertEqual(by_slice["case_a_slice0"]["metadata.volume_id"], "case_a")
        self.assertEqual(by_slice["case_a_slice1"]["metadata.slice_index"], "1")
        self.assertEqual(payload["threshold_analysis"]["oracle_per_case"]["case_count"], 2)
        self.assertEqual(
            payload["threshold_analysis"]["oracle_per_case"]["threshold_counts"],
            {"0.3": 1, "0.5": 1},
        )

    def test_2d_sweep_writes_global_threshold_without_oracle(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._make_2d_cfg(tmp, mode="sweep")
            result = self._run_mocked_2d_pipeline(cfg)
            payload = json.loads(Path(result["json_path"]).read_text())

        self.assertEqual(payload["threshold_analysis"]["best_global_threshold"]["threshold"], 0.5)
        self.assertIsNone(payload["threshold_analysis"]["oracle_per_case"])
        self.assertIsNone(result["oracle_csv_path"])

    def test_2d_oracle_mode_writes_oracle_without_global_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = self._make_2d_cfg(tmp, mode="oracle_per_case")
            result = self._run_mocked_2d_pipeline(cfg)
            payload = json.loads(Path(result["json_path"]).read_text())

        self.assertIsNone(payload["threshold_analysis"]["best_global_threshold"])
        self.assertIsNotNone(payload["threshold_analysis"]["oracle_per_case"])
        self.assertIsNotNone(result["oracle_csv_path"])

    def _make_2d_cfg(self, tmp: str, mode: str = "fixed"):
        cfg = _make_cfg(tmp, mode=mode)
        cfg.data_mode.dim = "2d"
        cfg.data_mode.loader_mode = "online_slices_3d_to_2d"
        cfg.evaluation.levels = ["slice"]
        cfg.evaluation.threshold_protocol.primary.level = "slice"
        cfg.evaluation.threshold_protocol.primary.metric = "Dice2DForegroundOnly"
        cfg.evaluation.metrics_2d = {"names": ["Dice2DForegroundOnly", "VoxelF1Score2D"]}
        return cfg

    def _run_mocked_pipeline(self, cfg):
        with patch(
            "scripts.evaluation.core.evaluation_pipeline.build_model_for_evaluation",
            return_value=DummyModel(),
        ), patch(
            "scripts.evaluation.core.evaluation_pipeline.get_dataloaders",
            return_value={"val": ["unused"]},
        ), patch(
            "scripts.evaluation.core.evaluation_pipeline.iter_model_volume_samples",
            return_value=iter([_volume_sample()]),
        ), patch(
            "scripts.evaluation.core.evaluation_pipeline.compute_metrics_3d_at_threshold",
            side_effect=_mock_metric_values,
        ):
            return run_model_evaluation(cfg)

    def _run_mocked_2d_pipeline(self, cfg, samples=None, metric_side_effect=None):
        if samples is None:
            samples = [("single_case", _slice_sample())]
        if metric_side_effect is None:
            metric_side_effect = _mock_slice_metric_values
        with patch(
            "scripts.evaluation.core.evaluation_pipeline.build_model_for_evaluation",
            return_value=DummyModel(),
        ), patch(
            "scripts.evaluation.core.evaluation_pipeline.get_dataloaders",
            return_value={"val": ["unused"]},
        ), patch(
            "scripts.evaluation.core.evaluation_pipeline.iter_diffusion_case_slice_samples",
            return_value=iter(samples),
        ), patch(
            "scripts.evaluation.core.evaluation_pipeline.compute_metrics_at_threshold",
            side_effect=metric_side_effect,
        ):
            return run_model_evaluation(cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
