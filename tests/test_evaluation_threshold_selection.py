"""
Tests for generic evaluation threshold selection helpers.
"""

import unittest

from scripts.evaluation.core.contracts import PrimaryMetricSelector
from scripts.evaluation.reporting.threshold_selection import select_best_threshold_from_rows


class TestThresholdSelection(unittest.TestCase):
    def test_select_max_mean_from_volume_rows(self):
        rows = [
            {
                "level": "volume",
                "threshold": 0.3,
                "metrics": {"DiceNativeCoefficient": {"mean": 0.7, "median": 0.69}},
            },
            {
                "level": "volume",
                "threshold": 0.5,
                "metrics": {"DiceNativeCoefficient": {"mean": 0.8, "median": 0.75}},
            },
        ]
        selector = PrimaryMetricSelector(
            level="volume",
            metric="DiceNativeCoefficient",
            statistic="mean",
            direction="max",
        )

        selected = select_best_threshold_from_rows(rows, selector)

        self.assertEqual(selected, 0.5)

    def test_select_max_median_from_volume_rows(self):
        rows = [
            {
                "level": "volume",
                "threshold": 0.3,
                "metrics": {"DiceNativeCoefficient": {"mean": 0.9, "median": 0.6}},
            },
            {
                "level": "volume",
                "threshold": 0.5,
                "metrics": {"DiceNativeCoefficient": {"mean": 0.8, "median": 0.7}},
            },
        ]
        selector = PrimaryMetricSelector(
            level="volume",
            metric="DiceNativeCoefficient",
            statistic="median",
            direction="max",
        )

        selected = select_best_threshold_from_rows(rows, selector)

        self.assertEqual(selected, 0.5)

    def test_select_min_mean_from_volume_rows(self):
        rows = [
            {
                "level": "volume",
                "threshold": 0.3,
                "metrics": {"HausdorffDistance95Native": {"mean": 12.0, "median": 10.0}},
            },
            {
                "level": "volume",
                "threshold": 0.5,
                "metrics": {"HausdorffDistance95Native": {"mean": 8.0, "median": 9.0}},
            },
        ]
        selector = PrimaryMetricSelector(
            level="volume",
            metric="HausdorffDistance95Native",
            statistic="mean",
            direction="min",
        )

        selected = select_best_threshold_from_rows(rows, selector)

        self.assertEqual(selected, 0.5)

    def test_ignores_rows_from_other_levels(self):
        rows = [
            {
                "level": "slice",
                "threshold": 0.3,
                "metrics": {"DiceNativeCoefficient": {"mean": 0.99}},
            },
            {
                "level": "volume",
                "threshold": 0.5,
                "metrics": {"DiceNativeCoefficient": {"mean": 0.5}},
            },
        ]
        selector = PrimaryMetricSelector(
            level="volume",
            metric="DiceNativeCoefficient",
            statistic="mean",
            direction="max",
        )

        selected = select_best_threshold_from_rows(rows, selector)

        self.assertEqual(selected, 0.5)

    def test_select_from_flattened_metric_columns(self):
        rows = [
            {"level": "volume", "threshold": 0.3, "DiceNativeCoefficient_mean": 0.7},
            {"level": "volume", "threshold": 0.5, "DiceNativeCoefficient_mean": 0.8},
        ]
        selector = PrimaryMetricSelector(
            level="volume",
            metric="DiceNativeCoefficient",
            statistic="mean",
            direction="max",
        )

        selected = select_best_threshold_from_rows(rows, selector)

        self.assertEqual(selected, 0.5)

    def test_select_from_current_scoped_slice_report_shape(self):
        rows = [
            {
                "level": "slice",
                "threshold": 0.3,
                "metrics": {
                    "dice": {
                        "all_slices": {"mean": 0.7},
                        "foreground_only": {"mean": 0.9},
                    }
                },
            },
            {
                "level": "slice",
                "threshold": 0.5,
                "metrics": {
                    "dice": {
                        "all_slices": {"mean": 0.8},
                        "foreground_only": {"mean": 0.85},
                    }
                },
            },
        ]
        selector = PrimaryMetricSelector(
            level="slice",
            metric="dice",
            statistic="mean",
            direction="max",
        )

        selected = select_best_threshold_from_rows(rows, selector)

        self.assertEqual(selected, 0.5)

    def test_empty_rows_raise(self):
        selector = PrimaryMetricSelector(
            level="volume",
            metric="DiceNativeCoefficient",
            statistic="mean",
            direction="max",
        )

        with self.assertRaises(ValueError):
            select_best_threshold_from_rows([], selector)

    def test_missing_metric_raises(self):
        rows = [{"level": "volume", "threshold": 0.5, "metrics": {}}]
        selector = PrimaryMetricSelector(
            level="volume",
            metric="DiceNativeCoefficient",
            statistic="mean",
            direction="max",
        )

        with self.assertRaises(ValueError):
            select_best_threshold_from_rows(rows, selector)


if __name__ == "__main__":
    unittest.main(verbosity=2)
