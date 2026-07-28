"""
Tests for canonical reporting helpers.
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluation.reporting.threshold_protocol import make_fixed_protocol, make_sweep_protocol
from scripts.evaluation.reporting import (
    append_per_slice_metrics_rows,
    build_report_payload,
    build_text_summary,
    write_json_report,
    write_threshold_csv,
    write_volume_threshold_csv,
)


def _make_finalized_results():
    return {
        0.5: {
            "threshold": 0.5,
            "slice_counts": {"total": 2, "foreground": 1, "empty": 1},
            "metrics": {
                "dice": {
                    "all_slices": {"mean": 0.9, "std": 0.1, "count": 2},
                    "foreground_only": {"mean": 0.95, "std": 0.05, "count": 1},
                },
                "precision": {
                    "all_slices": {"mean": 0.8, "std": 0.2, "count": 2},
                    "foreground_only": {"mean": 0.85, "std": 0.1, "count": 1},
                },
            },
        },
        0.7: {
            "threshold": 0.7,
            "slice_counts": {"total": 2, "foreground": 1, "empty": 1},
            "metrics": {
                "dice": {
                    "all_slices": {"mean": 0.91, "std": 0.09, "count": 2},
                    "foreground_only": {"mean": 0.96, "std": 0.04, "count": 1},
                },
                "precision": {
                    "all_slices": {"mean": 0.82, "std": 0.18, "count": 2},
                    "foreground_only": {"mean": 0.86, "std": 0.08, "count": 1},
                },
            },
        },
    }


class TestEvaluationReporting(unittest.TestCase):
    def test_build_report_payload(self):
        protocol = make_sweep_protocol([0.5, 0.7], optimize_metric="dice")
        payload = build_report_payload(
            finalized_results=_make_finalized_results(),
            protocol=protocol,
            entrypoint_name="compute_segmentation_metrics_for_diffusionmodel_2d_predictions",
            metadata={"run_id": "abc123"},
            selected_threshold=0.7,
            auc={"roc": 0.88, "pr": 0.55},
        )
        self.assertIn("metadata", payload)
        self.assertIn("data_summary", payload)
        self.assertIn("protocol", payload)
        self.assertIn("metrics", payload)
        self.assertIn("auc", payload)
        self.assertEqual(payload["protocol"]["selected_threshold"], 0.7)
        self.assertEqual(payload["metadata"]["run_id"], "abc123")
        self.assertIn("default_threshold_metrics", payload["metrics"])
        self.assertTrue(payload["metrics"]["default_threshold_metrics"]["evaluated"])
        self.assertEqual(payload["metrics"]["default_threshold_metrics"]["evaluated_threshold"], 0.5)

    def test_json_and_csv_writers(self):
        protocol = make_fixed_protocol(0.5)
        payload = build_report_payload(
            finalized_results={0.5: _make_finalized_results()[0.5]},
            protocol=protocol,
            entrypoint_name="compute_segmentation_metrics_for_nnunet_2d_predictions",
        )

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)

            json_path = write_json_report(payload, output_dir=out_dir)
            csv_path = write_threshold_csv(
                finalized_results={0.5: _make_finalized_results()[0.5]},
                output_dir=out_dir,
            )

            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())

            with json_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            self.assertEqual(
                loaded["metadata"]["entrypoint"],
                "compute_segmentation_metrics_for_nnunet_2d_predictions",
            )

            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                rows = list(reader)
            self.assertGreaterEqual(len(rows), 2)  # header + at least one row
            self.assertIn("threshold", rows[0])
            self.assertIn("dice_all_mean", rows[0])

    def test_append_per_slice_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics_per_slice.csv"
            fieldnames = ["case_id", "slice_id", "threshold", "dice"]

            append_per_slice_metrics_rows(
                rows=[
                    {"case_id": "a", "slice_id": "1", "threshold": 0.5, "dice": 0.8},
                    {"case_id": "b", "slice_id": "2", "threshold": 0.5, "dice": 0.7},
                ],
                output_csv_path=path,
                fieldnames=fieldnames,
            )
            append_per_slice_metrics_rows(
                rows=[
                    {"case_id": "c", "slice_id": "3", "threshold": 0.7, "dice": 0.9},
                ],
                output_csv_path=path,
                fieldnames=fieldnames,
            )

            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            self.assertEqual(len(rows), 3)

    def test_build_text_summary(self):
        payload = build_report_payload(
            finalized_results={0.5: _make_finalized_results()[0.5]},
            protocol=make_fixed_protocol(0.5),
            entrypoint_name="compute_segmentation_metrics_for_nnunet_2d_predictions",
        )
        summary = build_text_summary(payload)
        self.assertIn("Unified Segmentation Evaluation Summary", summary)
        self.assertIn("compute_segmentation_metrics_for_nnunet_2d_predictions", summary)

    def test_default_threshold_block_not_evaluated_when_missing(self):
        protocol = make_fixed_protocol(0.7)
        payload = build_report_payload(
            finalized_results={0.7: _make_finalized_results()[0.7]},
            protocol=protocol,
            entrypoint_name="compute_segmentation_metrics_for_diffusionmodel_2d_predictions",
            selected_threshold=0.7,
        )
        block = payload["metrics"]["default_threshold_metrics"]
        self.assertFalse(block["evaluated"])
        self.assertIsNone(block["evaluated_threshold"])
        self.assertIsNone(block["metrics"])

    def test_volume_level_payload_and_writer(self):
        slice_results = {0.5: _make_finalized_results()[0.5]}
        volume_results = {
            0.5: {
                "threshold": 0.5,
                "volume_counts": {"total": 2, "foreground": 1, "empty": 1},
                "volume_slice_counts": {
                    "mean": 3.5,
                    "std": 0.5,
                    "count": 2,
                    "total": 7,
                    "min": 3,
                    "max": 4,
                },
                "metrics": {
                    "DiceNativeCoefficient": {"mean": 0.9, "std": 0.1, "count": 2},
                },
            }
        }
        payload = build_report_payload(
            finalized_results=slice_results,
            protocol=make_fixed_protocol(0.5),
            entrypoint_name="compute_segmentation_metrics_for_nnunet_2d_predictions",
            volume_finalized_results=volume_results,
            selected_threshold=0.5,
        )
        self.assertIn("volume_level", payload["metrics"])
        self.assertEqual(payload["data_summary"]["total_volumes"], 2)
        self.assertEqual(payload["data_summary"]["volume_slice_count_total"], 7)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_volume_threshold_csv(volume_results, output_dir=Path(tmp))
            self.assertTrue(path.exists())
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertIn("total_volumes", rows[0])
            self.assertIn("DiceNativeCoefficient_mean", rows[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)

