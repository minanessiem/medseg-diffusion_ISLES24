"""
Tests for mask builder helpers in scripts.evaluation.
"""

import unittest

import torch

from scripts.evaluation.io.mask_builder import (
    build_ground_truth_mask,
    build_prediction_mask,
    ensure_binary_mask,
    threshold_probabilities,
)


class TestMaskBuilder(unittest.TestCase):
    def test_threshold_probabilities(self):
        pred = torch.tensor([[[0.9, 0.2], [0.5, 0.51]]])
        mask = threshold_probabilities(pred, threshold=0.5)
        expected = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        self.assertTrue(torch.equal(mask, expected))

    def test_ensure_binary_mask(self):
        mask = torch.tensor([[[2.0, 0.1], [0.0, 0.6]]])
        binary = ensure_binary_mask(mask)
        expected = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        self.assertTrue(torch.equal(binary, expected))

    def test_build_prediction_mask_from_probability(self):
        pred = torch.tensor([[[0.8, 0.1], [0.3, 0.9]]])
        mask = build_prediction_mask(prediction_prob=pred, threshold=0.5)
        expected = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        self.assertTrue(torch.equal(mask, expected))

    def test_build_prediction_mask_from_mask(self):
        pred_mask = torch.tensor([[[1.0, 0.0], [0.2, 0.8]]])
        mask = build_prediction_mask(prediction_mask=pred_mask)
        expected = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        self.assertTrue(torch.equal(mask, expected))

    def test_build_prediction_mask_requires_exactly_one_source(self):
        pred = torch.tensor([[[0.8]]])
        mask = torch.tensor([[[1.0]]])
        with self.assertRaises(ValueError):
            build_prediction_mask(prediction_prob=pred, prediction_mask=mask)
        with self.assertRaises(ValueError):
            build_prediction_mask()

    def test_build_ground_truth_mask(self):
        gt = torch.tensor([[[1.0, 0.0], [0.49, 0.51]]])
        binary = build_ground_truth_mask(gt)
        expected = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        self.assertTrue(torch.equal(binary, expected))


if __name__ == "__main__":
    unittest.main(verbosity=2)

