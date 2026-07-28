import unittest

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.losses.discriminative_deep_supervision import (
    DISCRIMINATIVE_LOSS_FACTORIES,
    compute_discriminative_deep_supervision_loss,
)
from src.losses.segmentation_losses import (
    DiceFocalLoss,
    FocalLoss,
    GeneralizedDiceLoss,
    HausdorffDTLoss,
    TverskyLoss,
)


try:
    import monai.losses  # noqa: F401

    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False


MONAI_LOSS_PARAMS = {
    "HausdorffDTLoss": {
        "alpha": 2.0,
        "include_background": True,
        "to_onehot_y": False,
        "sigmoid": False,
        "softmax": False,
        "reduction": "mean",
        "batch": False,
    },
    "TverskyLoss": {
        "include_background": True,
        "to_onehot_y": False,
        "sigmoid": False,
        "softmax": False,
        "alpha": 0.5,
        "beta": 0.5,
        "reduction": "mean",
        "smooth_nr": 1.0e-5,
        "smooth_dr": 1.0e-5,
        "batch": False,
        "soft_label": False,
    },
    "FocalLoss": {
        "include_background": True,
        "to_onehot_y": False,
        "gamma": 2.0,
        "alpha": None,
        "weight": None,
        "reduction": "mean",
        "use_softmax": False,
    },
    "DiceFocalLoss": {
        "include_background": True,
        "to_onehot_y": False,
        "sigmoid": True,
        "softmax": False,
        "squared_pred": False,
        "jaccard": False,
        "reduction": "mean",
        "smooth_nr": 1.0e-5,
        "smooth_dr": 1.0e-5,
        "batch": False,
        "gamma": 2.0,
        "weight": None,
        "lambda_dice": 1.0,
        "lambda_focal": 0.5,
        "alpha": None,
    },
    "GeneralizedDiceLoss": {
        "include_background": True,
        "to_onehot_y": False,
        "sigmoid": False,
        "softmax": False,
        "w_type": "square",
        "reduction": "mean",
        "smooth_nr": 1.0e-5,
        "smooth_dr": 1.0e-5,
        "batch": False,
        "soft_label": False,
    },
}


def _term(loss_key, supervision_mode):
    if supervision_mode == "inherit_default":
        supervision = {"mode": "inherit_default", "heads": None, "weights": None}
    else:
        supervision = {"mode": "final_only", "heads": [0], "weights": [1.0]}

    return {
        "loss": loss_key,
        "input_domain": "logits",
        "weight": 1.0,
        "params": {},
        "supervision": supervision,
    }


class _HeadRecordingLoss(nn.Module):
    def __init__(self, bucket, key):
        super().__init__()
        self.bucket = bucket
        self.key = key

    def forward(self, y_pred, y_true):
        self.bucket.setdefault(self.key, []).append(y_pred.detach().clone())
        return y_pred.sum() * 0.0 + y_true.new_tensor(1.0)


class TestDiscriminativeMonaiLosses(unittest.TestCase):
    def test_monai_loss_registry_has_explicit_required_params(self):
        for loss_key, params in MONAI_LOSS_PARAMS.items():
            self.assertIn(loss_key, DISCRIMINATIVE_LOSS_FACTORIES)

            incomplete_params = dict(params)
            missing_key = next(iter(incomplete_params.keys()))
            del incomplete_params[missing_key]

            with self.assertRaisesRegex(ValueError, f"requires params.{missing_key}"):
                DISCRIMINATIVE_LOSS_FACTORIES[loss_key](incomplete_params)

    @unittest.skipUnless(MONAI_AVAILABLE, "MONAI is not available in this environment.")
    def test_monai_loss_wrappers_return_finite_scalar_losses(self):
        logits = torch.tensor(
            [[[[-2.0, -1.0, 1.0, 2.0], [-2.0, 2.0, 2.0, -1.0]]]],
            dtype=torch.float32,
        )
        target = torch.tensor(
            [[[[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0]]]],
            dtype=torch.float32,
        )
        probabilities = torch.sigmoid(logits)

        cases = {
            "HausdorffDTLoss": (
                HausdorffDTLoss(**MONAI_LOSS_PARAMS["HausdorffDTLoss"]),
                probabilities,
            ),
            "TverskyLoss": (
                TverskyLoss(**MONAI_LOSS_PARAMS["TverskyLoss"]),
                probabilities,
            ),
            "FocalLoss": (FocalLoss(**MONAI_LOSS_PARAMS["FocalLoss"]), logits),
            "DiceFocalLoss": (
                DiceFocalLoss(**MONAI_LOSS_PARAMS["DiceFocalLoss"]),
                logits,
            ),
            "GeneralizedDiceLoss": (
                GeneralizedDiceLoss(**MONAI_LOSS_PARAMS["GeneralizedDiceLoss"]),
                probabilities,
            ),
        }

        for loss_key, (loss_fn, prediction) in cases.items():
            with self.subTest(loss_key=loss_key):
                loss = loss_fn(prediction, target)
                self.assertEqual(loss.dim(), 0)
                self.assertTrue(torch.isfinite(loss).item())

    def test_new_loss_can_be_final_head_only_with_all_head_anchor(self):
        target = torch.zeros(1, 1, 2, 2)
        final_head = torch.zeros(1, 1, 2, 2)
        auxiliary_head = torch.ones(1, 1, 2, 2)
        stacked_output = torch.stack([final_head, auxiliary_head], dim=1)
        captured = {}

        compute_discriminative_deep_supervision_loss(
            model_output=stacked_output,
            target=target,
            discriminative_cfg={
                "deep_supervision": {
                    "enabled": True,
                    "final_head": 0,
                    "head_parser": "stacked",
                    "default_supervision": {
                        "mode": "all_heads",
                        "weighting": "uniform",
                        "normalize": False,
                    },
                },
                "terms": [
                    _term("AnchorProbeLoss", "inherit_default"),
                    _term("BoundaryProbeLoss", "final_only"),
                ],
            },
            loss_factories={
                "AnchorProbeLoss": lambda params: _HeadRecordingLoss(
                    captured, "anchor"
                ),
                "BoundaryProbeLoss": lambda params: _HeadRecordingLoss(
                    captured, "boundary"
                ),
            },
        )

        self.assertEqual(len(captured["anchor"]), 2)
        self.assertEqual(len(captured["boundary"]), 1)
        self.assertTrue(torch.allclose(captured["anchor"][0], final_head))
        self.assertTrue(torch.allclose(captured["anchor"][1], auxiliary_head))
        self.assertTrue(torch.allclose(captured["boundary"][0], final_head))

    def test_deepsupervision_configs_keep_new_loss_terms_final_only(self):
        config_paths = [
            "configs/loss/discriminative_dice_logitsbce_hausdorffdt_deepsupervision.yaml",
            "configs/loss/discriminative_tversky_logitsbce_deepsupervision.yaml",
            "configs/loss/discriminative_dice_focal_deepsupervision.yaml",
            "configs/loss/discriminative_dicefocal_deepsupervision.yaml",
            "configs/loss/discriminative_generalizeddice_logitsbce_deepsupervision.yaml",
        ]
        new_loss_keys = set(MONAI_LOSS_PARAMS.keys())

        for config_path in config_paths:
            cfg = OmegaConf.load(config_path)
            for term in cfg.discriminative.terms:
                if term.loss in new_loss_keys:
                    self.assertEqual(term.supervision.mode, "final_only", config_path)
                    self.assertEqual(list(term.supervision.heads), [0], config_path)
                    self.assertEqual(list(term.supervision.weights), [1.0], config_path)


if __name__ == "__main__":
    unittest.main()
