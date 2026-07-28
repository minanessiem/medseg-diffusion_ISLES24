import unittest

from omegaconf import OmegaConf

from src.utils.run_name import generate_run_name


def _base_cfg():
    return OmegaConf.create(
        {
            "model": {
                "architecture": "medsegdiff",
                "image_size": 256,
                "num_layers": 4,
                "first_conv_channels": 16,
                "att_heads": 6,
                "att_head_dim": 4,
                "time_embedding_dim": 128,
                "bottleneck_transformer_layers": 1,
            },
            "data_runtime": {"train_batch_size": 4},
            "training": {
                "max_steps": 1000,
                "gradient": {"accumulation_steps": 1, "clip_norm": None},
                "amp": {"enabled": False, "dtype": "float32"},
            },
            "optimizer": {
                "optimizer_class": "adamw",
                "learning_rate": 1e-4,
                "weight_decay": 0.0,
            },
            "scheduler": {"scheduler_type": "constant"},
            "loss": {"loss_type": "MSE", "auxiliary_losses": {"enabled": False}},
            "diffusion": {
                "type": "OpenAI_DDPM",
                "sampling_mode": "ddpm",
                "timesteps": 1000,
                "noise_schedule": "linear",
                "timestep_respacing": "",
            },
            "dataset": {"modalities": ["CBF", "TMAX"], "num_modalities": 2},
            "data_mode": {
                "loader_mode": "nnunet_slices_2d",
                "per_side_context_slices": 2,
                "channel_layout": "modality_major",
            },
            "validation": {"ensemble": {"enabled": False, "num_samples": 1}},
        }
    )


class TestRunNameContextTokens(unittest.TestCase):
    def test_run_name_includes_nnunet_context_tokens(self):
        cfg = _base_cfg()
        run_name = generate_run_name(cfg, timestamp="2026-03-22_12-00-00")
        self.assertIn("ctxps2_mdmaj", run_name)

    def test_run_name_omits_context_tokens_for_non_nnunet_modes(self):
        cfg = _base_cfg()
        cfg.data_mode.loader_mode = "online_slices_3d_to_2d"
        run_name = generate_run_name(cfg, timestamp="2026-03-22_12-00-00")
        self.assertNotIn("ctxps", run_name)


if __name__ == "__main__":
    unittest.main()
