"""
Lightweight CLI and argument-validation tests for evaluation entrypoints.
"""

import unittest
from pathlib import Path

from scripts.evaluation.compute_segmentation_metrics_for_diffusionmodel_2d_predictions import (
    build_arg_parser as build_diffusion_parser,
    _build_analysis_cases,
    _parse_ensemble_samples,
    validate_args as validate_diffusion_args,
)
from scripts.evaluation.compute_segmentation_metrics_for_nnunet_2d_predictions import (
    build_arg_parser as build_nnunet_parser,
)


class TestEvaluationEntrypoints(unittest.TestCase):
    def test_nnunet_parser_defaults(self):
        parser = build_nnunet_parser()
        args = parser.parse_args(
            ["--pred-dir", "predictions", "--gt-dir", "labels"]
        )
        self.assertEqual(args.pred_dir, Path("predictions"))
        self.assertEqual(args.gt_dir, Path("labels"))
        self.assertEqual(args.fixed_threshold, 0.5)
        self.assertFalse(args.allow_shape_mismatch)

    def test_diffusion_parser_defaults(self):
        parser = build_diffusion_parser()
        args = parser.parse_args(
            ["--run-dir", "outputs/run_x", "--model-name", "best_model"]
        )
        self.assertEqual(args.run_dir, Path("outputs/run_x"))
        self.assertEqual(args.model_name, "best_model")
        self.assertEqual(args.ensemble_samples, "1")
        self.assertEqual(args.ensemble_method, "single")
        self.assertIsNone(args.fixed_threshold)
        self.assertFalse(args.test)
        self.assertEqual(args.test_max_slices, 10)

    def test_diffusion_validate_args_rejects_invalid_ensemble_combo(self):
        parser = build_diffusion_parser()

        args = parser.parse_args(
            [
                "--run-dir",
                "outputs/run_x",
                "--model-name",
                "best_model",
                "--ensemble-samples",
                "1,1",
                "--ensemble-method",
                "mean",
            ]
        )
        with self.assertRaises(ValueError):
            validate_diffusion_args(args)

        args = parser.parse_args(
            [
                "--run-dir",
                "outputs/run_x",
                "--model-name",
                "best_model",
                "--ensemble-samples",
                "3",
                "--ensemble-method",
                "single",
            ]
        )
        with self.assertRaises(ValueError):
            validate_diffusion_args(args)

    def test_parse_ensemble_samples_and_case_building(self):
        parser = build_diffusion_parser()
        parsed = _parse_ensemble_samples("1,3,3,5")
        self.assertEqual(parsed, [1, 3, 5])

        cases = _build_analysis_cases(parsed, "both")
        keys = {case["key"] for case in cases}
        self.assertIn("n1_single", keys)
        self.assertIn("n3_mean", keys)
        self.assertIn("n3_soft_staple", keys)
        self.assertIn("n5_mean", keys)
        self.assertIn("n5_soft_staple", keys)

        args = parser.parse_args(
            [
                "--run-dir",
                "outputs/run_x",
                "--model-name",
                "best_model",
                "--test-max-slices",
                "0",
            ]
        )
        with self.assertRaises(ValueError):
            validate_diffusion_args(args)


if __name__ == "__main__":
    unittest.main(verbosity=2)

