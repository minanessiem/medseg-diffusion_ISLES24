"""
Tests for per-case threshold record helpers.
"""

import csv
import math
import tempfile
import unittest
from pathlib import Path

from scripts.evaluation.core.contracts import PrimaryMetricSelector
from scripts.evaluation.reporting.threshold_records import (
    GROUND_TRUTH_VOLUME_KEY,
    PREDICTED_VOLUME_KEY,
    VOLUME_RATIO_KEY,
    ThresholdMetricRecord,
    add_volume_ratio,
    aggregate_threshold_records,
    select_global_threshold,
    select_oracle_thresholds,
    write_oracle_threshold_csv,
    write_per_case_threshold_csv,
)


def _record(case_id, threshold, dice, hd95=None, level="volume", metadata=None):
    metrics = {"DiceNativeCoefficient": float(dice)}
    if hd95 is not None:
        metrics["HausdorffDistance95Native"] = float(hd95)
    return ThresholdMetricRecord(
        level=level,
        case_id=case_id,
        threshold=float(threshold),
        metrics=metrics,
        metadata=metadata or {},
    )


def _slice_record(slice_id, threshold, dice, metadata=None):
    return ThresholdMetricRecord(
        level="slice",
        case_id=slice_id,
        threshold=float(threshold),
        metrics={"Dice2DForegroundOnly": float(dice)},
        metadata=metadata or {},
    )


class TestThresholdRecords(unittest.TestCase):
    def test_add_volume_ratio_normal_case(self):
        metrics = add_volume_ratio(
            {
                PREDICTED_VOLUME_KEY: 30.0,
                GROUND_TRUTH_VOLUME_KEY: 20.0,
            }
        )

        self.assertEqual(metrics[VOLUME_RATIO_KEY], 1.5)

    def test_add_volume_ratio_zero_gt_and_zero_prediction(self):
        metrics = add_volume_ratio(
            {
                PREDICTED_VOLUME_KEY: 0.0,
                GROUND_TRUTH_VOLUME_KEY: 0.0,
            }
        )

        self.assertEqual(metrics[VOLUME_RATIO_KEY], 1.0)

    def test_add_volume_ratio_zero_gt_and_nonzero_prediction(self):
        metrics = add_volume_ratio(
            {
                PREDICTED_VOLUME_KEY: 5.0,
                GROUND_TRUTH_VOLUME_KEY: 0.0,
            }
        )

        self.assertTrue(math.isinf(metrics[VOLUME_RATIO_KEY]))

    def test_aggregate_threshold_records_mean_median_std(self):
        records = [
            _record("case_a", 0.3, 0.2),
            _record("case_b", 0.3, 0.8),
            _record("case_a", 0.5, 0.4),
            _record("case_b", 0.5, 0.6),
            _record("slice_a", 0.5, 0.99, level="slice"),
        ]

        aggregates = aggregate_threshold_records(records, selector_level="volume")

        stats = aggregates[0.3]["metrics"]["DiceNativeCoefficient"]
        self.assertEqual(stats["count"], 2)
        self.assertAlmostEqual(stats["mean"], 0.5)
        self.assertAlmostEqual(stats["median"], 0.5)
        self.assertAlmostEqual(stats["std"], 0.3)
        self.assertEqual(aggregates[0.3]["case_count"], 2)
        self.assertNotIn(0.99, [stats["max"]])

    def test_select_global_threshold_max_mean(self):
        records = [
            _record("case_a", 0.3, 0.2),
            _record("case_b", 0.3, 0.8),
            _record("case_a", 0.5, 0.7),
            _record("case_b", 0.5, 0.9),
        ]
        selector = PrimaryMetricSelector(
            level="volume",
            metric="DiceNativeCoefficient",
            statistic="mean",
            direction="max",
        )

        result = select_global_threshold(records, selector)

        self.assertEqual(result["threshold"], 0.5)
        self.assertAlmostEqual(result["selected_statistic_value"], 0.8)

    def test_select_global_threshold_max_median(self):
        records = [
            _record("case_a", 0.3, 0.2),
            _record("case_b", 0.3, 0.9),
            _record("case_c", 0.3, 0.9),
            _record("case_a", 0.5, 0.7),
            _record("case_b", 0.5, 0.7),
            _record("case_c", 0.5, 0.7),
        ]
        selector = PrimaryMetricSelector(
            level="volume",
            metric="DiceNativeCoefficient",
            statistic="median",
            direction="max",
        )

        result = select_global_threshold(records, selector)

        self.assertEqual(result["threshold"], 0.3)
        self.assertAlmostEqual(result["selected_statistic_value"], 0.9)

    def test_select_global_threshold_min_mean(self):
        records = [
            _record("case_a", 0.3, 0.6, hd95=10.0),
            _record("case_b", 0.3, 0.7, hd95=12.0),
            _record("case_a", 0.5, 0.8, hd95=6.0),
            _record("case_b", 0.5, 0.9, hd95=8.0),
        ]
        selector = PrimaryMetricSelector(
            level="volume",
            metric="HausdorffDistance95Native",
            statistic="mean",
            direction="min",
        )

        result = select_global_threshold(records, selector)

        self.assertEqual(result["threshold"], 0.5)
        self.assertAlmostEqual(result["selected_statistic_value"], 7.0)

    def test_select_oracle_thresholds_per_case(self):
        records = [
            _record("case_a", 0.3, 0.2),
            _record("case_a", 0.5, 0.8),
            _record("case_b", 0.3, 0.9),
            _record("case_b", 0.5, 0.4),
        ]
        selector = PrimaryMetricSelector(
            level="volume",
            metric="DiceNativeCoefficient",
            statistic="mean",
            direction="max",
        )

        rows, summary = select_oracle_thresholds(records, selector)

        by_case = {row["case_id"]: row for row in rows}
        self.assertEqual(by_case["case_a"]["threshold"], 0.5)
        self.assertEqual(by_case["case_b"]["threshold"], 0.3)
        self.assertEqual(summary["case_count"], 2)
        self.assertAlmostEqual(
            summary["metrics"]["DiceNativeCoefficient"]["mean"],
            0.85,
        )

    def test_select_oracle_thresholds_per_slice_record(self):
        records = [
            _slice_record("case_a_slice0", 0.3, 0.9, metadata={"volume_id": "case_a"}),
            _slice_record("case_a_slice0", 0.5, 0.1, metadata={"volume_id": "case_a"}),
            _slice_record("case_a_slice1", 0.3, 0.2, metadata={"volume_id": "case_a"}),
            _slice_record("case_a_slice1", 0.5, 0.8, metadata={"volume_id": "case_a"}),
        ]
        selector = PrimaryMetricSelector(
            level="slice",
            metric="Dice2DForegroundOnly",
            statistic="mean",
            direction="max",
        )

        rows, summary = select_oracle_thresholds(records, selector)

        by_slice = {row["case_id"]: row for row in rows}
        self.assertEqual(float(by_slice["case_a_slice0"]["threshold"]), 0.3)
        self.assertEqual(float(by_slice["case_a_slice1"]["threshold"]), 0.5)
        self.assertEqual(summary["case_count"], 2)
        self.assertEqual(summary["threshold_counts"], {"0.3": 1, "0.5": 1})

    def test_write_per_case_threshold_csv_dynamic_columns(self):
        records = [
            ThresholdMetricRecord(
                level="volume",
                case_id="case_a",
                threshold=0.5,
                metrics={
                    "DiceNativeCoefficient": 0.8,
                    VOLUME_RATIO_KEY: math.inf,
                },
                metadata={"subset": "val_fast"},
            )
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = write_per_case_threshold_csv(records, Path(tmp))
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        self.assertIn("DiceNativeCoefficient", rows[0])
        self.assertIn(VOLUME_RATIO_KEY, rows[0])
        self.assertIn("metadata.subset", rows[0])
        self.assertEqual(rows[0][VOLUME_RATIO_KEY], "inf")

    def test_write_oracle_threshold_csv_dynamic_columns(self):
        oracle_rows = [
            {
                "level": "volume",
                "case_id": "case_a",
                "threshold": 0.5,
                "selected_metric": "DiceNativeCoefficient",
                "selected_statistic": "mean",
                "selected_value": 0.8,
                "metrics": {"DiceNativeCoefficient": 0.8},
                "metadata": {"subset": "val_fast"},
            }
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = write_oracle_threshold_csv(oracle_rows, Path(tmp))
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        self.assertIn("DiceNativeCoefficient", rows[0])
        self.assertIn("metadata.subset", rows[0])
        self.assertEqual(rows[0]["selected_metric"], "DiceNativeCoefficient")


if __name__ == "__main__":
    unittest.main(verbosity=2)
