import unittest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.diffusion.discriminative_adapter import DiscriminativeAdapter
from src.losses.discriminative_deep_supervision import (
    compute_discriminative_deep_supervision_loss,
    resolve_discriminative_terms,
)


def _term(loss_key, input_domain):
    return {
        "loss": loss_key,
        "input_domain": input_domain,
        "weight": 1.0,
        "params": {},
        "supervision": {
            "mode": "final_only",
            "heads": [0],
            "weights": [1.0],
        },
    }


def _discriminative_cfg(terms):
    return {
        "deep_supervision": {
            "enabled": False,
            "final_head": 0,
            "head_parser": "single",
        },
        "terms": terms,
    }


class _RecordingLoss(nn.Module):
    def __init__(self, bucket, key):
        super().__init__()
        self.bucket = bucket
        self.key = key

    def forward(self, y_pred, y_true):
        del y_true
        self.bucket[self.key] = y_pred.detach().clone()
        return y_pred.mean() * 0.0


class _FixedLogitModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", logits)
        self.image_channels = 1
        self.mask_channels = 1
        self.output_channels = 1
        self.image_size = int(logits.shape[-1])
        self.spatial_dims = 2

    def forward(self, x):
        batch_size = int(x.shape[0])
        return self.logits.expand(batch_size, -1, -1, -1)


class TestDiscriminativeOutputDomains(unittest.TestCase):
    def test_discriminative_terms_require_explicit_schema(self):
        with self.assertRaisesRegex(ValueError, "terms is required"):
            resolve_discriminative_terms({"deep_supervision": {"enabled": False}})

        with self.assertRaisesRegex(ValueError, "Legacy discriminative loss fields"):
            resolve_discriminative_terms(
                {
                    "deep_supervision": {"enabled": False},
                    "dice": {"enabled": True},
                    "terms": [_term("DiceLoss", "probabilities")],
                }
            )

        with self.assertRaisesRegex(ValueError, "unsupported field"):
            resolve_discriminative_terms(
                _discriminative_cfg(
                    [
                        {
                            **_term("DiceLoss", "probabilities"),
                            "enabled": True,
                        }
                    ]
                )
            )

        with self.assertRaisesRegex(ValueError, "input_domain"):
            resolve_discriminative_terms(
                _discriminative_cfg(
                    [
                        {
                            "loss": "DiceLoss",
                            "weight": 1.0,
                            "params": {},
                            "supervision": {
                                "mode": "final_only",
                                "heads": [0],
                                "weights": [1.0],
                            },
                        }
                    ]
                )
            )

    def test_discriminative_loss_routes_logits_and_probabilities(self):
        logits = torch.tensor([[[[-2.0, 0.0], [2.0, 4.0]]]], dtype=torch.float32)
        target = torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]], dtype=torch.float32)
        captured = {}

        result = compute_discriminative_deep_supervision_loss(
            model_output=logits,
            target=target,
            discriminative_cfg=_discriminative_cfg(
                [
                    _term("ProbabilityProbeLoss", "probabilities"),
                    _term("LogitProbeLoss", "logits"),
                ]
            ),
            loss_factories={
                "ProbabilityProbeLoss": lambda params: _RecordingLoss(
                    captured, "probabilities"
                ),
                "LogitProbeLoss": lambda params: _RecordingLoss(captured, "logits"),
            },
        )

        expected_probabilities = torch.sigmoid(logits)
        self.assertTrue(torch.allclose(captured["logits"], logits))
        self.assertTrue(torch.allclose(captured["probabilities"], expected_probabilities))
        self.assertTrue(torch.allclose(result.final_prediction, expected_probabilities))
        self.assertTrue(torch.allclose(result.head_predictions[0], expected_probabilities))

    def test_discriminative_adapter_sample_returns_probabilities(self):
        logits = torch.tensor([[[[-2.0, 0.0], [2.0, 4.0]]]], dtype=torch.float32)
        model = _FixedLogitModel(logits)
        cfg = OmegaConf.create(
            {
                "environment": {"device": "cpu"},
                "loss": {
                    "discriminative": _discriminative_cfg(
                        [
                            {
                                "loss": "DiceLoss",
                                "input_domain": "probabilities",
                                "weight": 1.0,
                                "params": {"smooth": 1.0e-5, "apply_sigmoid": False},
                                "supervision": {
                                    "mode": "final_only",
                                    "heads": [0],
                                    "weights": [1.0],
                                },
                            }
                        ]
                    )
                },
            }
        )
        adapter = DiscriminativeAdapter(model=model, cfg=cfg, device=torch.device("cpu"))

        conditioned_image = torch.zeros(1, 1, 2, 2)
        sample = adapter.sample(conditioned_image)
        snapshots = list(adapter.sample_with_snapshots(conditioned_image))

        expected = torch.sigmoid(logits)
        self.assertTrue(torch.all(sample >= 0.0))
        self.assertTrue(torch.all(sample <= 1.0))
        self.assertTrue(torch.allclose(sample, expected))
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0][0], 0)
        self.assertTrue(torch.allclose(snapshots[0][1], expected))

    def test_discriminative_loss_configs_use_explicit_active_terms(self):
        config_paths = [
            "configs/loss/discriminative_dicebce.yaml",
            "configs/loss/discriminative_dice_only.yaml",
            "configs/loss/discriminative_dicebce_deepsupervision.yaml",
            "configs/loss/discriminative_dice_only_deepsupervision.yaml",
            "configs/loss/discriminative_dice_logitsbce.yaml",
            "configs/loss/discriminative_dice_logitsbce_deepsupervision.yaml",
            "configs/loss/discriminative_dice_logitsbce_hausdorffdt.yaml",
            "configs/loss/discriminative_dice_logitsbce_hausdorffdt_deepsupervision.yaml",
            "configs/loss/discriminative_tversky_logitsbce.yaml",
            "configs/loss/discriminative_tversky_logitsbce_deepsupervision.yaml",
            "configs/loss/discriminative_dice_focal.yaml",
            "configs/loss/discriminative_dice_focal_deepsupervision.yaml",
            "configs/loss/discriminative_dicefocal.yaml",
            "configs/loss/discriminative_dicefocal_deepsupervision.yaml",
            "configs/loss/discriminative_generalizeddice_logitsbce.yaml",
            "configs/loss/discriminative_generalizeddice_logitsbce_deepsupervision.yaml",
        ]
        required_fields = {"loss", "input_domain", "weight", "params", "supervision"}

        for config_path in config_paths:
            cfg = OmegaConf.load(config_path)
            terms = list(cfg.discriminative.terms)
            self.assertTrue(
                terms, f"{config_path} must define at least one discriminative term."
            )
            for term in terms:
                term_dict = OmegaConf.to_container(term, resolve=True)
                self.assertNotIn("enabled", term_dict)
                self.assertNotIn("name", term_dict)
                self.assertTrue(required_fields.issubset(term_dict.keys()))
                self.assertIn(term_dict["input_domain"], {"logits", "probabilities"})

    def test_discriminative_loss_configs_use_conceptual_base_defaults(self):
        config_defaults = {
            "configs/loss/discriminative_dicebce.yaml": "discriminative_base",
            "configs/loss/discriminative_dice_only.yaml": "discriminative_base",
            "configs/loss/discriminative_dice_logitsbce.yaml": "discriminative_base",
            "configs/loss/discriminative_dice_logitsbce_hausdorffdt.yaml": "discriminative_base",
            "configs/loss/discriminative_tversky_logitsbce.yaml": "discriminative_base",
            "configs/loss/discriminative_dice_focal.yaml": "discriminative_base",
            "configs/loss/discriminative_dicefocal.yaml": "discriminative_base",
            "configs/loss/discriminative_generalizeddice_logitsbce.yaml": "discriminative_base",
            "configs/loss/discriminative_dicebce_deepsupervision.yaml": "discriminative_deepsupervision_base",
            "configs/loss/discriminative_dice_only_deepsupervision.yaml": "discriminative_deepsupervision_base",
            "configs/loss/discriminative_dice_logitsbce_deepsupervision.yaml": "discriminative_deepsupervision_base",
            "configs/loss/discriminative_dice_logitsbce_hausdorffdt_deepsupervision.yaml": "discriminative_deepsupervision_base",
            "configs/loss/discriminative_tversky_logitsbce_deepsupervision.yaml": "discriminative_deepsupervision_base",
            "configs/loss/discriminative_dice_focal_deepsupervision.yaml": "discriminative_deepsupervision_base",
            "configs/loss/discriminative_dicefocal_deepsupervision.yaml": "discriminative_deepsupervision_base",
            "configs/loss/discriminative_generalizeddice_logitsbce_deepsupervision.yaml": "discriminative_deepsupervision_base",
        }

        for config_path, expected_default in config_defaults.items():
            cfg = OmegaConf.load(config_path)
            defaults = OmegaConf.to_container(cfg.defaults, resolve=True)
            self.assertEqual(defaults, [expected_default], config_path)


if __name__ == "__main__":
    unittest.main()
