"""
Tests for dual-level streaming metrics engine.
"""

import unittest

import torch

from scripts.evaluation.core.contracts import SliceSample
from scripts.evaluation.metrics.engine import DualLevelStreamingMetricsEngine


class TestDualLevelStreamingMetricsEngine(unittest.TestCase):
    def test_streaming_volume_finalization_and_counts(self):
        engine = DualLevelStreamingMetricsEngine(
            thresholds=[0.5],
            volume_metric_names=["DiceNativeCoefficient"],
        )

        # Volume A with 2 slices, then volume B with 1 slice.
        samples = [
            SliceSample(
                case_id="sub-stroke0001",
                slice_id="sub-stroke0001_slice0",
                volume_id="sub-stroke0001",
                slice_index=0,
                prediction_prob=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
                ground_truth_mask=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
            ),
            SliceSample(
                case_id="sub-stroke0001",
                slice_id="sub-stroke0001_slice1",
                volume_id="sub-stroke0001",
                slice_index=1,
                prediction_prob=torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
                ground_truth_mask=torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
            ),
            SliceSample(
                case_id="sub-stroke0002",
                slice_id="sub-stroke0002_slice0",
                volume_id="sub-stroke0002",
                slice_index=0,
                prediction_prob=torch.tensor([[[0.0, 0.0], [0.0, 0.0]]]),
                ground_truth_mask=torch.tensor([[[0.0, 0.0], [0.0, 0.0]]]),
            ),
        ]

        finalized_midstream = engine.update(samples[0])
        self.assertEqual(len(finalized_midstream), 0)
        finalized_midstream = engine.update(samples[1])
        self.assertEqual(len(finalized_midstream), 0)

        # Volume boundary at next sample should flush first volume.
        finalized_midstream = engine.update(samples[2])
        self.assertEqual(len(finalized_midstream), 1)
        self.assertEqual(finalized_midstream[0].volume_id, "sub-stroke0001")
        self.assertEqual(finalized_midstream[0].metadata["num_slices"], 2)

        # Flush trailing open volume.
        trailing = engine.finalize_open_volumes()
        self.assertEqual(len(trailing), 1)
        self.assertEqual(trailing[0].volume_id, "sub-stroke0002")
        self.assertEqual(trailing[0].metadata["num_slices"], 1)

        finalized = engine.finalize()
        slice_result = finalized["slice_level"][0.5]
        volume_result = finalized["volume_level"][0.5]

        self.assertEqual(slice_result["slice_counts"]["total"], 3)
        self.assertEqual(volume_result["volume_counts"]["total"], 2)
        self.assertEqual(volume_result["volume_slice_counts"]["total"], 3)
        self.assertEqual(volume_result["volume_slice_counts"]["min"], 1)
        self.assertEqual(volume_result["volume_slice_counts"]["max"], 2)
        self.assertIn("DiceNativeCoefficient", volume_result["metrics"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
