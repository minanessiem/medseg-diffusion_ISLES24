import pytest
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


def test_discriminative_terms_require_explicit_schema():
    with pytest.raises(ValueError, match="terms is required"):
        resolve_discriminative_terms({"deep_supervision": {"enabled": False}})

    with pytest.raises(ValueError, match="Legacy discriminative loss fields"):
        resolve_discriminative_terms(
            {
                "deep_supervision": {"enabled": False},
                "dice": {"enabled": True},
                "terms": [_term("DiceLoss", "probabilities")],
            }
        )

    with pytest.raises(ValueError, match="unsupported field"):
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

    with pytest.raises(ValueError, match="input_domain"):
        resolve_discriminative_terms(
            _discriminative_cfg(
                [
                    {
                        "loss": "DiceLoss",
                        "weight": 1.0,
                        "params": {},
                        "supervision": {"mode": "final_only", "heads": [0], "weights": [1.0]},
                    }
                ]
            )
        )


def test_discriminative_loss_routes_logits_and_probabilities():
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
    assert torch.allclose(captured["logits"], logits)
    assert torch.allclose(captured["probabilities"], expected_probabilities)
    assert torch.allclose(result.final_prediction, expected_probabilities)
    assert torch.allclose(result.head_predictions[0], expected_probabilities)


def test_discriminative_adapter_sample_returns_probabilities():
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
    assert torch.all(sample >= 0.0)
    assert torch.all(sample <= 1.0)
    assert torch.allclose(sample, expected)
    assert len(snapshots) == 1
    assert snapshots[0][0] == 0
    assert torch.allclose(snapshots[0][1], expected)


def test_discriminative_loss_configs_use_explicit_active_terms():
    config_paths = [
        "configs/loss/discriminative_dicebce.yaml",
        "configs/loss/discriminative_dice_only.yaml",
        "configs/loss/discriminative_dicebce_deepsupervision.yaml",
        "configs/loss/discriminative_dice_only_deepsupervision.yaml",
    ]
    required_fields = {"loss", "input_domain", "weight", "params", "supervision"}

    for config_path in config_paths:
        cfg = OmegaConf.load(config_path)
        terms = list(cfg.discriminative.terms)
        assert terms, f"{config_path} must define at least one discriminative term."
        for term in terms:
            term_dict = OmegaConf.to_container(term, resolve=True)
            assert "enabled" not in term_dict
            assert "name" not in term_dict
            assert required_fields.issubset(term_dict.keys())
            assert term_dict["input_domain"] in {"logits", "probabilities"}
