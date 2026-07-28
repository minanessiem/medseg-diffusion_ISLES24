"""
Tests for evaluation provenance parsing and SliceSample volume fields.
"""

import unittest

import torch

from scripts.evaluation.core.contracts import SliceSample
from scripts.evaluation.io.provenance import (
    parse_diffusion_slice_identity,
    parse_nnunet_slice_identity,
)


class TestEvaluationProvenance(unittest.TestCase):
    def test_parse_diffusion_slice_identity(self):
        volume_id, slice_index = parse_diffusion_slice_identity("sub-stroke0001_slice37")
        self.assertEqual(volume_id, "sub-stroke0001")
        self.assertEqual(slice_index, 37)

    def test_parse_diffusion_slice_identity_with_full_path(self):
        volume_id, slice_index = parse_diffusion_slice_identity(
            r"C:\tmp\sub-stroke0002_slice0042.nii.gz"
        )
        self.assertEqual(volume_id, "sub-stroke0002")
        self.assertEqual(slice_index, 42)

    def test_parse_nnunet_slice_identity_label_name(self):
        volume_id, slice_index = parse_nnunet_slice_identity("sub-stroke0003_s0123.nii.gz")
        self.assertEqual(volume_id, "sub-stroke0003")
        self.assertEqual(slice_index, 123)

    def test_parse_nnunet_slice_identity_with_modality_suffix(self):
        volume_id, slice_index = parse_nnunet_slice_identity("sub-stroke0004_s0007_0001.nii.gz")
        self.assertEqual(volume_id, "sub-stroke0004")
        self.assertEqual(slice_index, 7)

    def test_parse_diffusion_slice_identity_rejects_invalid(self):
        with self.assertRaises(ValueError):
            parse_diffusion_slice_identity("sub-stroke0001_s0007")

    def test_parse_nnunet_slice_identity_rejects_invalid(self):
        with self.assertRaises(ValueError):
            parse_nnunet_slice_identity("sub-stroke0001_slice7")

    def test_slice_sample_validate_accepts_volume_fields(self):
        sample = SliceSample(
            case_id="legacy_case",
            slice_id="legacy_slice",
            volume_id="sub-stroke0010",
            slice_index=5,
            prediction_prob=torch.ones(1, 2, 2),
            ground_truth_mask=torch.ones(1, 2, 2),
        )
        sample.validate()

    def test_slice_sample_validate_rejects_partial_volume_fields(self):
        sample = SliceSample(
            case_id="legacy_case",
            slice_id="legacy_slice",
            volume_id="sub-stroke0011",
            prediction_prob=torch.ones(1, 2, 2),
            ground_truth_mask=torch.ones(1, 2, 2),
        )
        with self.assertRaises(ValueError):
            sample.validate()

    def test_slice_sample_validate_rejects_negative_slice_index(self):
        sample = SliceSample(
            case_id="legacy_case",
            slice_id="legacy_slice",
            volume_id="sub-stroke0012",
            slice_index=-1,
            prediction_prob=torch.ones(1, 2, 2),
            ground_truth_mask=torch.ones(1, 2, 2),
        )
        with self.assertRaises(ValueError):
            sample.validate()


if __name__ == "__main__":
    unittest.main(verbosity=2)
