"""
Tests for nnU-Net streaming IO producer.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.evaluation.io.nnunet import count_matched_pairs, iter_nnunet_slice_samples


class TestIoNnunet(unittest.TestCase):
    def test_streams_matched_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pred_dir = base / "pred"
            gt_dir = base / "gt"
            pred_dir.mkdir()
            gt_dir.mkdir()

            (pred_dir / "case_a_s0001.nii.gz").touch()
            (gt_dir / "case_a_s0001.nii.gz").touch()
            (gt_dir / "case_b_s0001.nii.gz").touch()

            def fake_loader(path: Path):
                if "case_a_s0001" in path.name:
                    tensor = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
                else:
                    tensor = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
                return tensor, torch.eye(4)

            with patch(
                "scripts.evaluation.io.nnunet._load_nifti_slice_tensor_with_affine",
                side_effect=fake_loader,
            ):
                samples = list(iter_nnunet_slice_samples(pred_dir=pred_dir, gt_dir=gt_dir))

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].case_id, "case_a")
            self.assertEqual(samples[0].slice_id, "case_a_s0001")
            self.assertEqual(samples[0].volume_id, "case_a")
            self.assertEqual(samples[0].slice_index, 1)
            self.assertEqual(tuple(samples[0].prediction_mask.shape), (1, 2, 2))

    def test_shape_mismatch_raises_when_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pred_dir = base / "pred"
            gt_dir = base / "gt"
            pred_dir.mkdir()
            gt_dir.mkdir()
            (pred_dir / "case_a_s0000.nii.gz").touch()
            (gt_dir / "case_a_s0000.nii.gz").touch()

            def fake_loader(path: Path):
                if "pred" in str(path.parent):
                    tensor = torch.zeros((1, 2, 2))
                else:
                    tensor = torch.zeros((1, 3, 3))
                return tensor, torch.eye(4)

            with patch(
                "scripts.evaluation.io.nnunet._load_nifti_slice_tensor_with_affine",
                side_effect=fake_loader,
            ):
                with self.assertRaises(ValueError):
                    list(iter_nnunet_slice_samples(pred_dir=pred_dir, gt_dir=gt_dir))

    def test_count_matched_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            pred_dir = base / "pred"
            gt_dir = base / "gt"
            pred_dir.mkdir()
            gt_dir.mkdir()
            (pred_dir / "case_a_s0000.nii.gz").touch()
            (pred_dir / "case_c_s0000.nii.gz").touch()
            (gt_dir / "case_a_s0000.nii.gz").touch()
            (gt_dir / "case_b_s0000.nii.gz").touch()

            matched, missing, total = count_matched_pairs(pred_dir=pred_dir, gt_dir=gt_dir)
            self.assertEqual(matched, 1)
            self.assertEqual(missing, 1)
            self.assertEqual(total, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)

