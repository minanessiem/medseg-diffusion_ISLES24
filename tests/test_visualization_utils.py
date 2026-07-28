import unittest

import torch
from omegaconf import OmegaConf

from src.utils.visualization_utils import (
    prepare_discriminative_sample_panels,
    prepare_discriminative_tensor_panel,
)


class TestVisualizationUtils(unittest.TestCase):
    def test_prepare_discriminative_sample_panels_2d(self):
        logging_cfg = OmegaConf.create(
            {
                "visualization": {
                    "mode": "triplanar",
                    "use_mip": False,
                }
            }
        )
        sample_img = torch.randn(2, 16, 20)
        sample_pred = torch.rand(1, 16, 20)
        sample_mask = torch.rand(1, 16, 20)
        modality_names = ["T1_RAW", "ADC_RAW"]

        panels, labels = prepare_discriminative_sample_panels(
            sample_img=sample_img,
            sample_pred=sample_pred,
            sample_mask=sample_mask,
            modality_names=modality_names,
            logging_cfg=logging_cfg,
        )

        self.assertEqual(len(panels), 4)
        self.assertEqual(len(labels), 4)
        self.assertEqual(panels[0].shape, (1, 16, 20))
        self.assertEqual(panels[-1].shape, (1, 16, 20))

    def test_prepare_discriminative_sample_panels_3d_triplanar_center(self):
        logging_cfg = OmegaConf.create(
            {
                "visualization": {
                    "mode": "triplanar",
                    "use_mip": False,
                }
            }
        )
        sample_img = torch.randn(2, 16, 20, 12)
        sample_pred = torch.rand(1, 16, 20, 12)
        sample_mask = torch.rand(1, 16, 20, 12)
        modality_names = ["T1_RAW", "ADC_RAW"]

        panels, labels = prepare_discriminative_sample_panels(
            sample_img=sample_img,
            sample_pred=sample_pred,
            sample_mask=sample_mask,
            modality_names=modality_names,
            logging_cfg=logging_cfg,
        )

        self.assertEqual(len(panels), 4)
        self.assertEqual(len(labels), 4)
        # After CCW display rotation: panel shape is [1, W, 3H].
        self.assertEqual(panels[0].shape, (1, 20, 48))
        self.assertEqual(panels[-1].shape, (1, 20, 48))

    def test_prepare_discriminative_sample_panels_3d_triplanar_mip(self):
        logging_cfg = OmegaConf.create(
            {
                "visualization": {
                    "mode": "triplanar",
                    "use_mip": True,
                }
            }
        )
        sample_img = torch.randn(1, 10, 14, 8)
        sample_pred = torch.rand(1, 10, 14, 8)
        sample_mask = torch.rand(1, 10, 14, 8)
        modality_names = ["T1_RAW"]

        panels, labels = prepare_discriminative_sample_panels(
            sample_img=sample_img,
            sample_pred=sample_pred,
            sample_mask=sample_mask,
            modality_names=modality_names,
            logging_cfg=logging_cfg,
        )

        self.assertEqual(len(panels), 3)
        self.assertEqual(len(labels), 3)
        self.assertEqual(panels[0].shape, (1, 14, 30))
        self.assertEqual(panels[-1].shape, (1, 14, 30))

    def test_prepare_discriminative_tensor_panel_3d(self):
        logging_cfg = OmegaConf.create(
            {
                "visualization": {
                    "mode": "triplanar",
                    "use_mip": False,
                }
            }
        )
        tensor_3d = torch.randn(1, 12, 18, 10)
        panel = prepare_discriminative_tensor_panel(
            tensor=tensor_3d,
            logging_cfg=logging_cfg,
        )
        self.assertEqual(panel.shape, (1, 18, 36))


if __name__ == "__main__":
    unittest.main(verbosity=2)
