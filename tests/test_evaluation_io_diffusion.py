"""
Tests for diffusion/custom streaming IO producer.
"""

import unittest

import torch

from scripts.evaluation.io.model_slices import (
    iter_diffusion_case_slice_samples,
    iter_diffusion_slice_samples,
)


class DummyDiffusionModel:
    def sample(self, image, disable_tqdm=True):
        del disable_tqdm
        return image * 0.5 + 0.25


class TestIoDiffusion(unittest.TestCase):
    def test_single_sample_streaming(self):
        model = DummyDiffusionModel()
        image = torch.tensor(
            [
                [[[1.0, 0.0], [0.5, 0.1]]],
                [[[0.1, 0.2], [0.3, 0.4]]],
            ]
        )
        mask = torch.tensor(
            [
                [[[1.0, 0.0], [1.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 0.0]]],
            ]
        )
        paths = ["case_a_slice1", "case_b_slice9"]
        dataloader = [(image, mask, paths)]

        samples = list(
            iter_diffusion_slice_samples(
                model=model,
                dataloader=dataloader,
                device="cpu",
                ensemble_num_samples=1,
                ensemble_method="single",
            )
        )

        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].case_id, "case_a")
        self.assertEqual(samples[1].slice_id, "case_b_slice9")
        self.assertEqual(samples[0].volume_id, "case_a")
        self.assertEqual(samples[0].slice_index, 1)
        self.assertEqual(samples[1].volume_id, "case_b")
        self.assertEqual(samples[1].slice_index, 9)
        self.assertTrue(torch.all(samples[0].prediction_prob >= 0.0))
        self.assertTrue(torch.all(samples[0].prediction_prob <= 1.0))

    def test_mean_ensemble_requires_num_samples(self):
        model = DummyDiffusionModel()
        dataloader = [(torch.zeros((1, 1, 2, 2)), torch.zeros((1, 1, 2, 2)), None)]
        with self.assertRaises(ValueError):
            list(
                iter_diffusion_slice_samples(
                    model=model,
                    dataloader=dataloader,
                    device="cpu",
                    ensemble_num_samples=1,
                    ensemble_method="mean",
                )
            )

    def test_mean_ensemble_streaming(self):
        model = DummyDiffusionModel()
        image = torch.tensor([[[[0.8, 0.2], [0.4, 0.6]]]])
        mask = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        dataloader = [(image, mask)]

        samples = list(
            iter_diffusion_slice_samples(
                model=model,
                dataloader=dataloader,
                device="cpu",
                ensemble_num_samples=2,
                ensemble_method="mean",
            )
        )
        self.assertEqual(len(samples), 1)
        self.assertIsNotNone(samples[0].prediction_prob)
        self.assertEqual(samples[0].metadata["ensemble_method"], "mean")
        self.assertEqual(samples[0].metadata["ensemble_num_samples"], 2)

    def test_max_samples_limits_stream(self):
        model = DummyDiffusionModel()
        image = torch.tensor(
            [
                [[[1.0, 0.0], [0.5, 0.1]]],
                [[[0.1, 0.2], [0.3, 0.4]]],
            ]
        )
        mask = torch.tensor(
            [
                [[[1.0, 0.0], [1.0, 0.0]]],
                [[[0.0, 0.0], [0.0, 0.0]]],
            ]
        )
        dataloader = [(image, mask)]

        samples = list(
            iter_diffusion_slice_samples(
                model=model,
                dataloader=dataloader,
                device="cpu",
                ensemble_num_samples=1,
                ensemble_method="single",
                show_progress=False,
                max_samples=1,
            )
        )
        self.assertEqual(len(samples), 1)

    def test_multi_case_reuses_sample_stack_behavior(self):
        model = DummyDiffusionModel()
        image = torch.tensor([[[[0.8, 0.2], [0.4, 0.6]]]])
        mask = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        dataloader = [(image, mask, ["case_x_slice0"])]
        analysis_cases = [
            {"key": "n1_single", "method": "single", "num_samples": 1},
            {"key": "n2_mean", "method": "mean", "num_samples": 2},
        ]

        yielded = list(
            iter_diffusion_case_slice_samples(
                model=model,
                dataloader=dataloader,
                device="cpu",
                analysis_cases=analysis_cases,
                max_requested_size=2,
                show_progress=False,
                max_samples=1,
            )
        )
        case_keys = [case_key for case_key, _ in yielded]
        self.assertEqual(case_keys, ["n1_single", "n2_mean"])
        self.assertEqual(yielded[0][1].case_id, "case_x")
        self.assertEqual(yielded[0][1].volume_id, "case_x")
        self.assertEqual(yielded[0][1].slice_index, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

