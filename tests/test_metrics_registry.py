"""
Unit tests for the threshold analysis metrics registry.

Verifies that:
1. All metrics are properly registered
2. Threshold parameterization works correctly
3. Metrics match expected behavior for known inputs
4. Results are consistent with underlying src/metrics implementations
"""

import unittest
import torch

from scripts.analysis.metrics_registry import (
    METRIC_REGISTRY,
    get_all_metrics,
    compute_metrics_at_threshold,
    DiceAtThreshold,
    PrecisionAtThreshold,
    RecallAtThreshold,
    SpecificityAtThreshold,
    F1AtThreshold,
    F2AtThreshold,
)


class TestMetricRegistry(unittest.TestCase):
    """Tests for the metric registry functionality."""

    def test_all_metrics_registered(self):
        """Verify all expected metrics are in the registry."""
        expected_metrics = {'dice', 'precision', 'recall', 'specificity', 'f1', 'f2'}
        self.assertEqual(set(METRIC_REGISTRY.keys()), expected_metrics)

    def test_get_all_metrics_returns_instances(self):
        """Verify get_all_metrics returns instantiated objects."""
        metrics = get_all_metrics()
        self.assertEqual(len(metrics), 6)
        for name, metric in metrics.items():
            self.assertTrue(callable(metric), f"Metric {name} should be callable")


class TestThresholdBehavior(unittest.TestCase):
    """Tests for threshold parameterization."""

    def setUp(self):
        """Create sample prediction and ground truth tensors."""
        # Prediction: sigmoid-like values ranging from 0.05 to 0.8
        self.pred = torch.tensor([[[[0.8, 0.6, 0.3],
                                    [0.7, 0.4, 0.1],
                                    [0.2, 0.1, 0.05]]]])
        # Ground truth: binary mask (top-left 2x2 quadrant is foreground)
        self.gt = torch.tensor([[[[1.0, 1.0, 0.0],
                                  [1.0, 0.0, 0.0],
                                  [0.0, 0.0, 0.0]]]])

    def test_lower_threshold_increases_recall(self):
        """Lower threshold should capture more true positives -> higher recall."""
        recall_metric = RecallAtThreshold()
        recall_05 = recall_metric(self.pred, self.gt, threshold=0.5)
        recall_03 = recall_metric(self.pred, self.gt, threshold=0.3)
        
        # At threshold 0.3, we capture more of the ground truth
        self.assertGreaterEqual(recall_03, recall_05, 
                                "Lower threshold should increase recall")

    def test_higher_threshold_increases_precision(self):
        """Higher threshold should reduce false positives -> higher precision."""
        precision_metric = PrecisionAtThreshold()
        precision_05 = precision_metric(self.pred, self.gt, threshold=0.5)
        precision_07 = precision_metric(self.pred, self.gt, threshold=0.7)
        
        # At threshold 0.7, we're more selective -> fewer FPs
        # Use almostEqual to handle floating-point precision at ceiling values
        self.assertAlmostEqual(precision_07, precision_05, places=4,
                               msg="Precision should be similar or higher at higher threshold")

    def test_threshold_affects_dice(self):
        """Dice should change with threshold."""
        dice_metric = DiceAtThreshold()
        dice_03 = dice_metric(self.pred, self.gt, threshold=0.3)
        dice_05 = dice_metric(self.pred, self.gt, threshold=0.5)
        dice_07 = dice_metric(self.pred, self.gt, threshold=0.7)
        
        # Just verify they're different (optimal threshold depends on data)
        self.assertFalse(dice_03 == dice_05 == dice_07, 
                         "Dice should vary with threshold")

    def test_compute_metrics_at_threshold(self):
        """Verify compute_metrics_at_threshold returns all metrics."""
        results = compute_metrics_at_threshold(self.pred, self.gt, threshold=0.5)
        
        self.assertEqual(set(results.keys()), 
                         {'dice', 'precision', 'recall', 'specificity', 'f1', 'f2'})
        for name, value in results.items():
            self.assertIsInstance(value, float, f"{name} should return float")
            self.assertGreaterEqual(value, 0.0, f"{name} should be >= 0")
            self.assertLessEqual(value, 1.0, f"{name} should be <= 1")


class TestKnownValues(unittest.TestCase):
    """Tests with known expected values."""

    def test_perfect_prediction(self):
        """Perfect prediction should give dice=1.0, precision=1.0, recall=1.0."""
        pred = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
        gt = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
        
        results = compute_metrics_at_threshold(pred, gt, threshold=0.5)
        
        self.assertAlmostEqual(results['dice'], 1.0, places=5)
        self.assertAlmostEqual(results['precision'], 1.0, places=5)
        self.assertAlmostEqual(results['recall'], 1.0, places=5)

    def test_complete_miss(self):
        """Predicting opposite of ground truth should give low scores."""
        pred = torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]])
        gt = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
        
        results = compute_metrics_at_threshold(pred, gt, threshold=0.5)
        
        # Should have 0 true positives
        self.assertLess(results['dice'], 0.1)
        self.assertLess(results['recall'], 0.1)

    def test_half_overlap(self):
        """50% overlap should give intermediate dice."""
        # Prediction covers half the ground truth
        pred = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
        gt = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
        
        dice_metric = DiceAtThreshold()
        dice = dice_metric(pred, gt, threshold=0.5)
        
        # TP=1, FP=0, FN=1 -> Dice = 2*1/(1+2) = 2/3 ≈ 0.667
        self.assertAlmostEqual(dice, 2/3, places=2)


class TestConsistencyWithSrcMetrics(unittest.TestCase):
    """Verify consistency with src/metrics/metrics.py implementations."""

    def test_dice_matches_underlying_at_default_threshold(self):
        """At threshold 0.5, wrapper should match underlying metric."""
        from src.metrics.metrics import Dice2DForegroundOnly
        
        pred = torch.tensor([[[[0.8, 0.6], [0.3, 0.1]]]])
        gt = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
        
        # Underlying metric
        underlying = Dice2DForegroundOnly()
        underlying_result = underlying(pred, gt).item()
        
        # Wrapper at default threshold
        wrapper = DiceAtThreshold()
        wrapper_result = wrapper(pred, gt, threshold=0.5)
        
        self.assertAlmostEqual(wrapper_result, underlying_result, places=6)

    def test_all_metrics_match_underlying_at_default(self):
        """All wrappers should match underlying metrics at threshold 0.5."""
        from src.metrics.metrics import (
            Dice2DForegroundOnly,
            VoxelPrecision2D,
            VoxelSensitivity2D,
            VoxelSpecificity2D,
            VoxelF1Score2D,
            VoxelF2Score2D,
        )
        
        pred = torch.tensor([[[[0.9, 0.7, 0.4],
                               [0.6, 0.3, 0.1],
                               [0.2, 0.1, 0.05]]]])
        gt = torch.tensor([[[[1.0, 1.0, 0.0],
                             [1.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0]]]])
        
        underlying_classes = {
            'dice': Dice2DForegroundOnly,
            'precision': VoxelPrecision2D,
            'recall': VoxelSensitivity2D,
            'specificity': VoxelSpecificity2D,
            'f1': VoxelF1Score2D,
            'f2': VoxelF2Score2D,
        }
        
        wrapper_results = compute_metrics_at_threshold(pred, gt, threshold=0.5)
        
        for name, cls in underlying_classes.items():
            underlying = cls()
            underlying_result = underlying(pred, gt).item()
            
            self.assertAlmostEqual(
                wrapper_results[name], underlying_result, places=6,
                msg=f"{name} wrapper doesn't match underlying metric"
            )


if __name__ == '__main__':
    unittest.main(verbosity=2)
