"""
Tests for greenfield evaluation metrics registry parity.
"""

import unittest

import torch

from scripts.analysis.metrics_registry import (
    METRIC_REGISTRY as LEGACY_METRIC_REGISTRY,
    compute_metrics_at_threshold as legacy_compute_metrics_at_threshold,
)
from scripts.evaluation.metrics.registry_2d import (
    METRIC_REGISTRY as EVAL_METRIC_REGISTRY,
    compute_metrics_at_threshold as eval_compute_metrics_at_threshold,
    resolve_2d_metric_class_names,
)


class TestEvaluationMetricRegistryParity(unittest.TestCase):
    """Ensure the greenfield registry stays aligned with existing analysis behavior."""

    def test_metric_registry_uses_class_name_keys(self):
        self.assertEqual(
            set(EVAL_METRIC_REGISTRY.keys()),
            {
                "Dice2DForegroundOnly",
                "VoxelPrecision2D",
                "VoxelSensitivity2D",
                "VoxelSpecificity2D",
                "VoxelF1Score2D",
                "VoxelF2Score2D",
            },
        )

    def test_legacy_aliases_resolve_to_class_names(self):
        self.assertEqual(
            resolve_2d_metric_class_names(LEGACY_METRIC_REGISTRY.keys()),
            (
                "Dice2DForegroundOnly",
                "VoxelPrecision2D",
                "VoxelSensitivity2D",
                "VoxelSpecificity2D",
                "VoxelF1Score2D",
                "VoxelF2Score2D",
            ),
        )

    def test_threshold_results_parity_with_analysis_registry(self):
        pred = torch.tensor([[[[0.9, 0.7, 0.2], [0.8, 0.1, 0.1], [0.4, 0.2, 0.05]]]])
        gt = torch.tensor([[[[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]])
        threshold = 0.5

        legacy = legacy_compute_metrics_at_threshold(pred, gt, threshold)
        greenfield = eval_compute_metrics_at_threshold(pred, gt, threshold)

        alias_to_class = dict(
            zip(LEGACY_METRIC_REGISTRY.keys(), resolve_2d_metric_class_names(LEGACY_METRIC_REGISTRY.keys()))
        )
        self.assertEqual(set(alias_to_class.values()), set(greenfield.keys()))
        for metric_name, class_name in alias_to_class.items():
            self.assertAlmostEqual(
                legacy[metric_name],
                greenfield[class_name],
                places=6,
                msg=f"Metric mismatch for {metric_name}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

