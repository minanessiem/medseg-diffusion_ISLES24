"""
Tests for the streaming evaluation metrics engine.
"""

import unittest

import torch

from scripts.evaluation.core.contracts import SliceSample
from scripts.evaluation.metrics.engine import StreamingMetricsEngine, get_scope_metric


class TestStreamingMetricsEngine(unittest.TestCase):
    """Validate scope-aware streaming accumulation behavior."""

    def _build_samples(self):
        # Sample 1: foreground slice, strong overlap.
        s1 = SliceSample(
            case_id="case_001",
            slice_id="0",
            prediction_prob=torch.tensor([[[0.9, 0.8], [0.2, 0.1]]]),
            ground_truth_mask=torch.tensor([[[1.0, 1.0], [0.0, 0.0]]]),
        )
        # Sample 2: empty slice, prediction mostly empty.
        s2 = SliceSample(
            case_id="case_001",
            slice_id="1",
            prediction_prob=torch.tensor([[[0.1, 0.1], [0.2, 0.2]]]),
            ground_truth_mask=torch.tensor([[[0.0, 0.0], [0.0, 0.0]]]),
        )
        return [s1, s2]

    def test_counts_and_scopes(self):
        engine = StreamingMetricsEngine(thresholds=[0.5])
        results = engine.run(self._build_samples())

        threshold_result = results[0.5]
        self.assertEqual(threshold_result["slice_counts"]["total"], 2)
        self.assertEqual(threshold_result["slice_counts"]["foreground"], 1)
        self.assertEqual(threshold_result["slice_counts"]["empty"], 1)

        # Foreground-only scope should count only the foreground sample.
        dice_fg = get_scope_metric(results, 0.5, "Dice2DForegroundOnly", "foreground_only")
        self.assertEqual(dice_fg["count"], 1)

        # All-slices scope should count both samples.
        precision_all = get_scope_metric(results, 0.5, "VoxelPrecision2D", "all_slices")
        self.assertEqual(precision_all["count"], 2)

    def test_mask_input_mode_supports_post_threshold_sources(self):
        engine = StreamingMetricsEngine(thresholds=[0.5])
        sample = SliceSample(
            case_id="nn_case_01",
            slice_id="3",
            prediction_mask=torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
            ground_truth_mask=torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
        )
        results = engine.run([sample])

        dice_all = get_scope_metric(results, 0.5, "Dice2DForegroundOnly", "all_slices")
        self.assertEqual(dice_all["count"], 1)
        self.assertAlmostEqual(dice_all["mean"], 1.0, places=6)

    def test_metric_name_aliases_resolve_to_class_names(self):
        engine = StreamingMetricsEngine(thresholds=[0.5], metric_names=["dice_2d_fg", "f1_2d"])
        results = engine.run(self._build_samples())

        metric_names = set(results[0.5]["metrics"].keys())
        self.assertEqual(metric_names, {"Dice2DForegroundOnly", "VoxelF1Score2D"})


if __name__ == "__main__":
    unittest.main(verbosity=2)

