import unittest

import hydra

from src.utils.run_name import generate_run_name


class TestDynUNetPhase5Profiles(unittest.TestCase):
    """Phase 5 checks: DynUNet profiles compose and run-name supports DynUNet/DS."""

    def _compose(self, config_name, overrides=None):
        with hydra.initialize(config_path="../configs", version_base=None):
            return hydra.compose(config_name=config_name, overrides=overrides or [])

    def test_local_dynunet_profile_composes_and_generates_run_name(self):
        cfg = self._compose(
            "local_isles26_3d_randompatch_dynunet",
            overrides=[
                "training.max_steps=1",
                "data_runtime.train_batch_size=1",
            ],
        )
        self.assertEqual(cfg.model.architecture, "dynunet")
        self.assertTrue(cfg.model.deep_supervision)
        run_name = generate_run_name(cfg, timestamp="2026-05-30_00-00-00")
        self.assertIn("dynunet_", run_name)
        self.assertIn("dynunet_64_3d_k3-3-3-3_f32-64-128-256", run_name)
        self.assertNotIn("p1n1", run_name)
        self.assertNotIn("ampBF16", run_name)
        self.assertIn("dsup2", run_name)
        self.assertNotIn("_s1-2-2-2_", run_name)
        self.assertNotIn("_u2-2-2_", run_name)
        self.assertNotIn("wd00", run_name)
        self.assertNotIn("clip", run_name)

    def test_cluster_dynunet_profile_composes_and_generates_run_name(self):
        cfg = self._compose(
            "cluster_isles26_3d_randompatch_dynunet",
            overrides=[
                "training.max_steps=1",
                "data_runtime.train_batch_size=1",
            ],
        )
        self.assertEqual(cfg.model.architecture, "dynunet")
        self.assertTrue(cfg.model.deep_supervision)
        run_name = generate_run_name(cfg, timestamp="2026-05-30_00-00-00")
        self.assertIn("dynunet_", run_name)
        self.assertIn("dynunet_128_3d_k3-3-3-3_f32-64-128-256", run_name)
        self.assertNotIn("p1n1", run_name)
        self.assertNotIn("ampBF16", run_name)
        self.assertIn("dsup2", run_name)
        self.assertNotIn("_s1-2-2-2_", run_name)
        self.assertNotIn("_u2-2-2_", run_name)
        self.assertNotIn("wd00", run_name)
        self.assertNotIn("clip", run_name)

    def test_run_name_includes_dsup_token_when_enabled(self):
        cfg = self._compose(
            "local_isles26_3d_randompatch_dynunet",
            overrides=[
                "model.deep_supervision=true",
                "training.max_steps=1",
                "data_runtime.train_batch_size=1",
            ],
        )
        run_name = generate_run_name(cfg, timestamp="2026-05-30_00-00-00")
        self.assertIn("dynunet_", run_name)
        self.assertIn("dsup2", run_name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
