"""
Tests for threshold calibration CSV digest helpers.
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_threshold_calibration import (
    build_case_summary_rows,
    parse_volume_bins,
    read_threshold_metrics_csv,
    resolve_metric_names,
    select_global_threshold,
    write_digest_outputs,
)


def _write_sample_csv(path: Path) -> None:
    fieldnames = [
        "level",
        "case_id",
        "threshold",
        "DiceNativeCoefficient",
        "GroundTruthVolumeMm3",
        "PredictedVolumeMm3",
        "pred_gt_volume_ratio",
        "VoxelFalsePositives",
        "VoxelFalseNegatives",
        "metadata.volume_id",
    ]
    rows = [
        ("case_a", 0.5, 0.50, 100.0, 80.0, 0.8, 10, 30),
        ("case_a", 0.65, 0.70, 100.0, 90.0, 0.9, 8, 18),
        ("case_a", 0.8, 0.60, 100.0, 60.0, 0.6, 4, 35),
        ("case_b", 0.5, 0.60, 6000.0, 9000.0, 1.5, 80, 40),
        ("case_b", 0.65, 0.65, 6000.0, 7200.0, 1.2, 50, 45),
        ("case_b", 0.8, 0.90, 6000.0, 6200.0, 1.033333, 20, 10),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case_id, threshold, dice, gt, pred, ratio, fp, fn in rows:
            writer.writerow(
                {
                    "level": "volume",
                    "case_id": case_id,
                    "threshold": threshold,
                    "DiceNativeCoefficient": dice,
                    "GroundTruthVolumeMm3": gt,
                    "PredictedVolumeMm3": pred,
                    "pred_gt_volume_ratio": ratio,
                    "VoxelFalsePositives": fp,
                    "VoxelFalseNegatives": fn,
                    "metadata.volume_id": case_id,
                }
            )


class TestSummarizeThresholdCalibration(unittest.TestCase):
    def test_build_case_summary_selects_global_and_oracle_thresholds(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_csv = Path(tmp) / "per_case_threshold_metrics.csv"
            _write_sample_csv(input_csv)

            rows = read_threshold_metrics_csv(input_csv, level="volume")
            metric_names = resolve_metric_names(
                rows,
                metrics_spec="DiceNativeCoefficient,GroundTruthVolumeMm3,VoxelFalsePositives,VoxelFalseNegatives",
                selector_metric="DiceNativeCoefficient",
            )
            global_selection = select_global_threshold(
                rows=rows,
                metric="DiceNativeCoefficient",
                statistic="mean",
                direction="max",
                override_threshold=None,
                threshold_tolerance=1.0e-6,
            )

            self.assertEqual(global_selection.threshold, 0.8)

            case_rows = build_case_summary_rows(
                rows=rows,
                metric_names=metric_names,
                selector_metric="DiceNativeCoefficient",
                fixed_threshold=0.5,
                global_threshold=global_selection.threshold,
                direction="max",
                threshold_tolerance=1.0e-6,
                gt_volume_column="GroundTruthVolumeMm3",
                volume_bins=parse_volume_bins("0,1000,10000,inf"),
                significant_gain=0.1,
                allow_missing_cases=False,
            )
            by_case = {row["case_id"]: row for row in case_rows}

            self.assertEqual(by_case["case_a"]["oracle_threshold"], 0.65)
            self.assertEqual(by_case["case_b"]["oracle_threshold"], 0.8)
            self.assertAlmostEqual(
                by_case["case_a"]["oracle_improvement_over_fixed"],
                0.20,
            )
            self.assertEqual(by_case["case_a"]["gt_volume_bin"], "[0,1000)")
            self.assertEqual(by_case["case_b"]["gt_volume_bin"], "[1000,10000)")

    def test_write_digest_outputs_creates_expected_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_csv = Path(tmp) / "per_case_threshold_metrics.csv"
            output_dir = Path(tmp) / "digest"
            _write_sample_csv(input_csv)

            rows = read_threshold_metrics_csv(input_csv, level="volume")
            metric_names = resolve_metric_names(
                rows,
                metrics_spec="all",
                selector_metric="DiceNativeCoefficient",
            )
            global_selection = select_global_threshold(
                rows=rows,
                metric="DiceNativeCoefficient",
                statistic="mean",
                direction="max",
                override_threshold=0.65,
                threshold_tolerance=1.0e-6,
            )
            case_rows = build_case_summary_rows(
                rows=rows,
                metric_names=metric_names,
                selector_metric="DiceNativeCoefficient",
                fixed_threshold=0.5,
                global_threshold=global_selection.threshold,
                direction="max",
                threshold_tolerance=1.0e-6,
                gt_volume_column="GroundTruthVolumeMm3",
                volume_bins=parse_volume_bins("0,1000,10000,inf"),
                significant_gain=0.1,
                allow_missing_cases=False,
            )

            outputs = write_digest_outputs(
                output_dir=output_dir,
                case_rows=case_rows,
                global_candidates=global_selection.candidates,
                metric_names=metric_names,
                selector_metric="DiceNativeCoefficient",
                fixed_threshold=0.5,
                global_threshold=global_selection.threshold,
                global_statistic="mean",
                direction="max",
                significant_gain=0.1,
                gt_volume_column="GroundTruthVolumeMm3",
                volume_bins=parse_volume_bins("0,1000,10000,inf"),
                input_csv=input_csv,
            )

            self.assertTrue(outputs.case_summary_csv.exists())
            self.assertTrue(outputs.metric_summary_csv.exists())
            self.assertTrue(outputs.global_candidates_csv.exists())
            self.assertTrue(outputs.volume_bin_summary_csv.exists())
            self.assertTrue(outputs.oracle_threshold_distribution_csv.exists())
            summary = json.loads(outputs.json_summary.read_text(encoding="utf-8"))
            self.assertEqual(summary["case_count"], 2)
            self.assertEqual(summary["thresholds"]["global"], 0.65)
            self.assertEqual(
                summary["significant_gain"]["oracle_gain_case_count"],
                2,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
