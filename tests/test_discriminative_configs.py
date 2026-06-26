"""
Verification test for discriminative configs (Phase 5.6).

Run with: python3 -m unittest tests.test_discriminative_configs
"""

import unittest
from pathlib import Path

from omegaconf import OmegaConf


def _term_losses(cfg):
    return [term.loss for term in cfg.discriminative.terms]


def _load_loss_config(config_name):
    return _compose_loss_config(Path("configs/loss") / f"{config_name}.yaml")


def _compose_loss_config(config_path):
    cfg = OmegaConf.load(config_path)
    defaults = list(cfg.pop("defaults", [])) if "defaults" in cfg else []

    merged = OmegaConf.create()
    for entry in defaults:
        if entry == "_self_":
            continue
        if not isinstance(entry, str):
            raise TypeError(f"Unsupported loss config default entry: {entry!r}")
        merged = OmegaConf.merge(
            merged,
            _compose_loss_config(config_path.parent / f"{entry}.yaml"),
        )

    return OmegaConf.merge(merged, cfg)


class TestDiscriminativeConfigs(unittest.TestCase):
    def test_individual_configs(self):
        """Test that individual config files load correctly."""
        print("=" * 60)
        print("Discriminative Config Verification Test")
        print("=" * 60)

        print("\n[1/7] Testing configs/diffusion/discriminative.yaml...")
        cfg_diff = OmegaConf.load('configs/diffusion/discriminative.yaml')
        self.assertEqual(cfg_diff.type, 'Discriminative')
        self.assertEqual(cfg_diff.timesteps, 1)
        print(f"  ✓ type: {cfg_diff.type}")
        print(f"  ✓ timesteps: {cfg_diff.timesteps}")

        print("\n[2/7] Testing configs/model/swinunetr_base.yaml...")
        cfg_model = OmegaConf.load('configs/model/swinunetr_base.yaml')
        self.assertEqual(cfg_model.architecture, 'swinunetr')
        self.assertEqual(cfg_model.image_size, 256)
        self.assertEqual(cfg_model.feature_size, 48)
        print(f"  ✓ architecture: {cfg_model.architecture}")
        print(f"  ✓ image_size: {cfg_model.image_size}")
        print(f"  ✓ feature_size: {cfg_model.feature_size}")
        print(f"  ✓ depths: {cfg_model.depths}")
        print(f"  ✓ num_heads: {cfg_model.num_heads}")

        print("\n[3/7] Testing configs/loss/discriminative_dicebce.yaml...")
        cfg_loss = _load_loss_config('discriminative_dicebce')
        self.assertEqual(cfg_loss.loss_type, 'NONE')
        self.assertFalse(cfg_loss.discriminative.deep_supervision.enabled)
        self.assertEqual(_term_losses(cfg_loss), ['DiceLoss', 'BCELoss'])
        self.assertEqual(cfg_loss.discriminative.terms[0].input_domain, 'probabilities')
        self.assertEqual(cfg_loss.discriminative.terms[1].input_domain, 'probabilities')
        self.assertFalse('dice' in cfg_loss.discriminative)
        self.assertFalse('bce' in cfg_loss.discriminative)
        print(f"  ✓ loss_type: {cfg_loss.loss_type}")
        print(f"  ✓ discriminative.terms: {_term_losses(cfg_loss)}")

        print("\n[4/7] Testing configs/loss/discriminative_dice_only.yaml...")
        cfg_dice_only = _load_loss_config('discriminative_dice_only')
        self.assertFalse(cfg_dice_only.discriminative.deep_supervision.enabled)
        self.assertEqual(_term_losses(cfg_dice_only), ['DiceLoss'])
        self.assertFalse('dice' in cfg_dice_only.discriminative)
        self.assertFalse('bce' in cfg_dice_only.discriminative)
        print(f"  ✓ discriminative.terms: {_term_losses(cfg_dice_only)}")

        print("\n[5/7] Testing configs/loss/discriminative_dicebce_deepsupervision.yaml...")
        cfg_dicebce_ds = _load_loss_config('discriminative_dicebce_deepsupervision')
        self.assertTrue(cfg_dicebce_ds.discriminative.deep_supervision.enabled)
        self.assertEqual(cfg_dicebce_ds.discriminative.deep_supervision.final_head, 0)
        self.assertEqual(cfg_dicebce_ds.discriminative.deep_supervision.head_parser, 'stacked')
        self.assertEqual(_term_losses(cfg_dicebce_ds), ['DiceLoss', 'BCELoss'])
        print(f"  ✓ discriminative.deep_supervision.enabled: {cfg_dicebce_ds.discriminative.deep_supervision.enabled}")
        print(f"  ✓ discriminative.terms: {_term_losses(cfg_dicebce_ds)}")

        print("\n[6/7] Testing configs/loss/discriminative_dice_only_deepsupervision.yaml...")
        cfg_dice_only_ds = _load_loss_config('discriminative_dice_only_deepsupervision')
        self.assertTrue(cfg_dice_only_ds.discriminative.deep_supervision.enabled)
        self.assertEqual(cfg_dice_only_ds.discriminative.deep_supervision.final_head, 0)
        self.assertEqual(cfg_dice_only_ds.discriminative.deep_supervision.head_parser, 'stacked')
        self.assertEqual(_term_losses(cfg_dice_only_ds), ['DiceLoss'])
        print(f"  ✓ discriminative.deep_supervision.enabled: {cfg_dice_only_ds.discriminative.deep_supervision.enabled}")
        print(f"  ✓ discriminative.terms: {_term_losses(cfg_dice_only_ds)}")

        print("\n[7/7] Testing configs/logging/2d_discriminative.yaml...")
        cfg_log = OmegaConf.load('configs/logging/2d_discriminative.yaml')
        self.assertFalse(cfg_log.enable_sampling_snapshots)
        self.assertEqual(cfg_log.sampling_log_interval, 0)
        print(f"  ✓ enable_sampling_snapshots: {cfg_log.enable_sampling_snapshots}")
        print(f"  ✓ sampling_log_interval: {cfg_log.sampling_log_interval}")
        print(f"  ✓ enable_image_logging: {cfg_log.enable_image_logging}")

        print("\n" + "=" * 60)
        print("✓ All discriminative config tests passed!")
        print("=" * 60)


if __name__ == "__main__":
    unittest.main()

