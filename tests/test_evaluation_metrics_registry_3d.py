"""
Tests for 3D evaluation metrics registry.
"""

import unittest

import torch

from scripts.evaluation.metrics.registry_3d import (
    THREED_METRIC_CLASSES,
    compute_metrics_3d_at_threshold,
    get_all_metrics_3d,
)


class TestEvaluationMetricsRegistry3D(unittest.TestCase):
    def test_registry_uses_class_name_keys(self):
        metrics = get_all_metrics_3d()
        self.assertEqual(set(metrics.keys()), set(THREED_METRIC_CLASSES.keys()))

    def test_all_3d_metrics_compute_scalar_values(self):
        pred = torch.tensor(
            [
                [
                    [[0.9, 0.1], [0.0, 0.8]],
                    [[0.0, 0.2], [0.1, 0.9]],
                ]
            ],
            dtype=torch.float32,
        )  # [C, H, W, D]
        gt = torch.tensor(
            [
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[0.0, 0.0], [0.0, 1.0]],
                ]
            ],
            dtype=torch.float32,
        )  # [C, H, W, D]

        results = compute_metrics_3d_at_threshold(pred=pred, gt=gt, threshold=0.5)
        self.assertEqual(set(results.keys()), set(THREED_METRIC_CLASSES.keys()))
        for metric_name, value in results.items():
            self.assertIsInstance(value, float, msg=f"{metric_name} did not return float.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
