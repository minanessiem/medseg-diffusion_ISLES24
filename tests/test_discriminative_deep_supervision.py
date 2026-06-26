import unittest

import torch
import torch.nn as nn

from src.losses.discriminative_deep_supervision import (
    compute_discriminative_deep_supervision_loss,
    normalize_discriminative_head_outputs,
)


class _L1Loss(nn.Module):
    def forward(self, pred, target):
        return torch.abs(pred - target).mean()


def _term(loss_key, input_domain, weight=1.0, params=None, supervision=None):
    return {
        "loss": loss_key,
        "input_domain": input_domain,
        "weight": weight,
        "params": params or {},
        "supervision": supervision or {"mode": "final_only"},
    }


def _dice_term(supervision=None):
    return _term(
        "DiceLoss",
        "probabilities",
        params={"smooth": 1.0e-5, "apply_sigmoid": False},
        supervision=supervision,
    )


class TestDiscriminativeDeepSupervision(unittest.TestCase):
    def test_normalize_single_output(self):
        pred = torch.rand(2, 1, 8, 8)
        heads = normalize_discriminative_head_outputs(pred, head_parser="single")
        self.assertEqual(sorted(heads.keys()), [0])
        self.assertEqual(tuple(heads[0].shape), (2, 1, 8, 8))

    def test_normalize_stacked_output(self):
        pred = torch.rand(2, 3, 1, 8, 8)
        heads = normalize_discriminative_head_outputs(pred, head_parser="stacked")
        self.assertEqual(sorted(heads.keys()), [0, 1, 2])
        self.assertEqual(tuple(heads[0].shape), (2, 1, 8, 8))
        self.assertEqual(tuple(heads[2].shape), (2, 1, 8, 8))

    def test_normalize_stacked_output_3d(self):
        pred = torch.rand(1, 2, 1, 8, 8, 8)
        heads = normalize_discriminative_head_outputs(pred, head_parser="stacked")
        self.assertEqual(sorted(heads.keys()), [0, 1])
        self.assertEqual(tuple(heads[0].shape), (1, 1, 8, 8, 8))
        self.assertEqual(tuple(heads[1].shape), (1, 1, 8, 8, 8))

    def test_weighted_head_aggregation(self):
        target = torch.zeros(1, 1, 8, 8)
        # head0: perfect prediction -> loss 0
        # head1: all ones vs zeros -> L1 loss 1
        pred = torch.stack(
            [torch.zeros(1, 1, 8, 8), torch.ones(1, 1, 8, 8)],
            dim=1,
        )  # [B,S,C,H,W] => [1,2,1,8,8]

        cfg = {
            "deep_supervision": {
                "enabled": True,
                "final_head": 0,
                "head_parser": "stacked",
                "default_supervision": {
                    "mode": "weighted_heads",
                    "heads": [0, 1],
                    "weights": [1.0, 0.5],
                },
            },
            "terms": [
                _term(
                    "L1Loss",
                    "logits",
                    weight=2.0,
                    supervision={"mode": "inherit_default"},
                )
            ],
        }

        result = compute_discriminative_deep_supervision_loss(
            model_output=pred,
            target=target,
            discriminative_cfg=cfg,
            loss_factories={"L1Loss": lambda _params: _L1Loss()},
        )

        # weighted_head_sum = 1.0*0 + 0.5*1 = 0.5
        # term_total = term_weight(2.0) * 0.5 = 1.0
        self.assertAlmostEqual(result.loss_components["L1Loss_head0"], 0.0, places=6)
        self.assertAlmostEqual(result.loss_components["L1Loss_head1"], 1.0, places=6)
        self.assertAlmostEqual(result.loss_components["L1Loss"], 1.0, places=6)
        self.assertAlmostEqual(result.loss_components["total"], 1.0, places=6)
        self.assertEqual(tuple(result.final_prediction.shape), (1, 1, 8, 8))

    def test_weighted_head_aggregation_3d(self):
        target = torch.zeros(1, 1, 8, 8, 8)
        pred = torch.stack(
            [torch.zeros(1, 1, 8, 8, 8), torch.ones(1, 1, 8, 8, 8)],
            dim=1,
        )  # [1,2,1,8,8,8]

        cfg = {
            "deep_supervision": {
                "enabled": True,
                "final_head": 0,
                "head_parser": "stacked",
                "default_supervision": {
                    "mode": "weighted_heads",
                    "heads": [0, 1],
                    "weights": [1.0, 0.5],
                },
            },
            "terms": [
                _term(
                    "L1Loss",
                    "logits",
                    weight=2.0,
                    supervision={"mode": "inherit_default"},
                )
            ],
        }

        result = compute_discriminative_deep_supervision_loss(
            model_output=pred,
            target=target,
            discriminative_cfg=cfg,
            loss_factories={"L1Loss": lambda _params: _L1Loss()},
        )

        self.assertAlmostEqual(result.loss_components["L1Loss_head0"], 0.0, places=6)
        self.assertAlmostEqual(result.loss_components["L1Loss_head1"], 1.0, places=6)
        self.assertAlmostEqual(result.loss_components["L1Loss"], 1.0, places=6)
        self.assertAlmostEqual(result.loss_components["total"], 1.0, places=6)
        self.assertEqual(tuple(result.final_prediction.shape), (1, 1, 8, 8, 8))

    def test_auto_parser_single_3d_output(self):
        """Auto parser should keep 3D single-head tensor as single output."""
        pred = torch.rand(1, 1, 8, 8, 8)   # [B,C,H,W,D]
        target = torch.rand(1, 1, 8, 8, 8) # same rank as pred
        cfg = {
            "deep_supervision": {"enabled": False, "final_head": 0, "head_parser": "auto"},
            "terms": [_dice_term()],
        }

        result = compute_discriminative_deep_supervision_loss(
            model_output=pred,
            target=target,
            discriminative_cfg=cfg,
        )
        self.assertEqual(tuple(result.final_prediction.shape), (1, 1, 8, 8, 8))

    def test_stacked_parser_falls_back_to_single_when_eval_output_is_single(self):
        """
        DynUNet deep supervision returns stacked heads only during training.
        In eval mode it returns a single tensor; parser must handle this.
        """
        pred = torch.rand(1, 1, 8, 8, 8)   # single-head eval-style output
        target = torch.rand(1, 1, 8, 8, 8)
        cfg = {
            "deep_supervision": {
                "enabled": True,
                "final_head": 0,
                "head_parser": "stacked",
                "default_supervision": {
                    "mode": "all_heads",
                    "weighting": "geometric",
                    "decay": 0.5,
                },
            },
            "terms": [
                _dice_term(supervision={"mode": "inherit_default"})
            ],
        }

        result = compute_discriminative_deep_supervision_loss(
            model_output=pred,
            target=target,
            discriminative_cfg=cfg,
        )
        self.assertIn("DiceLoss_head0", result.loss_components)
        self.assertNotIn("DiceLoss_head1", result.loss_components)
        self.assertEqual(tuple(result.final_prediction.shape), (1, 1, 8, 8, 8))

    def test_all_heads_geometric_weighting(self):
        target = torch.zeros(1, 1, 8, 8)
        pred = torch.stack(
            [
                torch.full((1, 1, 8, 8), 1.0),
                torch.full((1, 1, 8, 8), 2.0),
                torch.full((1, 1, 8, 8), 4.0),
            ],
            dim=1,
        )  # [1,3,1,8,8]

        cfg = {
            "deep_supervision": {
                "enabled": True,
                "final_head": 0,
                "head_parser": "stacked",
                "default_supervision": {
                    "mode": "all_heads",
                    "weighting": "geometric",
                    "decay": 0.5,
                },
            },
            "terms": [
                _term(
                    "L1Loss",
                    "logits",
                    supervision={"mode": "inherit_default"},
                )
            ],
        }

        result = compute_discriminative_deep_supervision_loss(
            model_output=pred,
            target=target,
            discriminative_cfg=cfg,
            loss_factories={"L1Loss": lambda _params: _L1Loss()},
        )

        # L1 values are 1,2,4. Geometric weights with decay=0.5: [1.0,0.5,0.25]
        # Total = 1*1 + 0.5*2 + 0.25*4 = 3.0
        self.assertAlmostEqual(result.loss_components["L1Loss_head0"], 1.0, places=6)
        self.assertAlmostEqual(result.loss_components["L1Loss_head1"], 2.0, places=6)
        self.assertAlmostEqual(result.loss_components["L1Loss_head2"], 4.0, places=6)
        self.assertAlmostEqual(result.loss_components["L1Loss"], 3.0, places=6)
        self.assertAlmostEqual(result.loss_components["total"], 3.0, places=6)

    def test_shape_mismatch_raises(self):
        target = torch.zeros(1, 1, 8, 8)
        pred = torch.zeros(1, 1, 4, 4)

        cfg = {
            "deep_supervision": {"enabled": False, "final_head": 0, "head_parser": "single"},
            "terms": [
                _term("L1Loss", "logits", supervision={"mode": "final_only"})
            ],
        }

        with self.assertRaisesRegex(ValueError, "Shape mismatch"):
            compute_discriminative_deep_supervision_loss(
                model_output=pred,
                target=target,
                discriminative_cfg=cfg,
                loss_factories={"L1Loss": lambda _params: _L1Loss()},
            )

    def test_invalid_final_head_raises(self):
        target = torch.zeros(1, 1, 8, 8)
        pred = torch.rand(1, 2, 1, 8, 8)

        cfg = {
            "deep_supervision": {
                "enabled": True,
                "final_head": 3,
                "head_parser": "stacked",
                "default_supervision": {
                    "mode": "weighted_heads",
                    "heads": [0, 1],
                    "weights": [1.0, 0.5],
                },
            },
            "terms": [
                _term(
                    "L1Loss",
                    "logits",
                    supervision={"mode": "inherit_default"},
                )
            ],
        }

        with self.assertRaisesRegex(ValueError, "final_head=3"):
            compute_discriminative_deep_supervision_loss(
                model_output=pred,
                target=target,
                discriminative_cfg=cfg,
                loss_factories={"L1Loss": lambda _params: _L1Loss()},
            )

    def test_legacy_dice_bce_config_raises(self):
        pred = torch.rand(1, 1, 8, 8)
        target = torch.rand(1, 1, 8, 8)

        cfg = {
            "deep_supervision": {"enabled": False, "final_head": 0, "head_parser": "single"},
            "dice": {
                "enabled": True,
                "weight": 1.0,
                "smooth": 1.0e-5,
                "apply_sigmoid": False,
            },
            "bce": {
                "enabled": False,
                "weight": 0.0,
                "pos_weight": None,
                "apply_sigmoid": False,
            },
        }

        with self.assertRaisesRegex(ValueError, "Legacy discriminative loss fields"):
            compute_discriminative_deep_supervision_loss(
                model_output=pred,
                target=target,
                discriminative_cfg=cfg,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
