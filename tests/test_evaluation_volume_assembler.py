"""
Tests for volume assembly from slice streams.
"""

import unittest

import torch

from scripts.evaluation.core.contracts import SliceSample
from scripts.evaluation.io.volume_assembler import VolumeAssembler


class TestVolumeAssembler(unittest.TestCase):
    def _sample(
        self,
        *,
        case_id: str,
        volume_id: str,
        slice_index: int,
        pred_value: float,
        gt_value: float,
    ) -> SliceSample:
        return SliceSample(
            case_id=case_id,
            slice_id=f"{volume_id}_slice{slice_index}",
            volume_id=volume_id,
            slice_index=slice_index,
            prediction_prob=torch.full((1, 2, 2), fill_value=pred_value, dtype=torch.float32),
            ground_truth_mask=torch.full((1, 2, 2), fill_value=gt_value, dtype=torch.float32),
        )

    def test_finalize_volume_orders_by_slice_index(self):
        assembler = VolumeAssembler()
        # Add out of order intentionally: 2, 0, 1
        assembler.add_sample(
            "n3_mean",
            self._sample(
                case_id="sub-stroke0001", volume_id="sub-stroke0001", slice_index=2, pred_value=2.0, gt_value=20.0
            ),
        )
        assembler.add_sample(
            "n3_mean",
            self._sample(
                case_id="sub-stroke0001", volume_id="sub-stroke0001", slice_index=0, pred_value=0.0, gt_value=0.0
            ),
        )
        assembler.add_sample(
            "n3_mean",
            self._sample(
                case_id="sub-stroke0001", volume_id="sub-stroke0001", slice_index=1, pred_value=1.0, gt_value=10.0
            ),
        )

        vol = assembler.finalize_volume("n3_mean", "sub-stroke0001")
        self.assertIsNotNone(vol)
        self.assertEqual(tuple(vol.prediction_volume.shape), (1, 2, 2, 3))
        # D-axis values should follow sorted indices: 0,1,2
        self.assertAlmostEqual(float(vol.prediction_volume[0, 0, 0, 0]), 0.0)
        self.assertAlmostEqual(float(vol.prediction_volume[0, 0, 0, 1]), 1.0)
        self.assertAlmostEqual(float(vol.prediction_volume[0, 0, 0, 2]), 2.0)

    def test_case_isolation_for_same_volume_id(self):
        assembler = VolumeAssembler()
        assembler.add_sample(
            "n1_single",
            self._sample(
                case_id="sub-stroke0002", volume_id="sub-stroke0002", slice_index=0, pred_value=1.0, gt_value=1.0
            ),
        )
        assembler.add_sample(
            "n5_soft_staple",
            self._sample(
                case_id="sub-stroke0002", volume_id="sub-stroke0002", slice_index=0, pred_value=5.0, gt_value=1.0
            ),
        )
        vol_single = assembler.finalize_volume("n1_single", "sub-stroke0002")
        vol_staple = assembler.finalize_volume("n5_soft_staple", "sub-stroke0002")
        self.assertIsNotNone(vol_single)
        self.assertIsNotNone(vol_staple)
        self.assertAlmostEqual(float(vol_single.prediction_volume[0, 0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(vol_staple.prediction_volume[0, 0, 0, 0]), 5.0)

    def test_duplicate_slice_index_raises(self):
        assembler = VolumeAssembler()
        first = self._sample(
            case_id="sub-stroke0003", volume_id="sub-stroke0003", slice_index=4, pred_value=0.0, gt_value=0.0
        )
        second = self._sample(
            case_id="sub-stroke0003", volume_id="sub-stroke0003", slice_index=4, pred_value=1.0, gt_value=1.0
        )
        assembler.add_sample("n1_single", first)
        with self.assertRaises(ValueError):
            assembler.add_sample("n1_single", second)

    def test_finalize_all_drains_buffers(self):
        assembler = VolumeAssembler()
        assembler.add_sample(
            "n1_single",
            self._sample(
                case_id="sub-stroke0010", volume_id="sub-stroke0010", slice_index=0, pred_value=0.0, gt_value=0.0
            ),
        )
        assembler.add_sample(
            "n1_single",
            self._sample(
                case_id="sub-stroke0011", volume_id="sub-stroke0011", slice_index=0, pred_value=1.0, gt_value=1.0
            ),
        )
        self.assertEqual(assembler.buffer_size(), 2)
        grouped = assembler.finalize_all()
        self.assertIn("n1_single", grouped)
        self.assertEqual(len(grouped["n1_single"]), 2)
        self.assertEqual(assembler.buffer_size(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
