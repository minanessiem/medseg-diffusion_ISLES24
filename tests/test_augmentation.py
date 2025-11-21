"""
Unit tests for data augmentation pipeline.

Tests cover:
- Transform building (none, light, aggressive)
- Augmentation application (modify image, preserve label structure)
- Deterministic behavior (same seed = same output)

Run with: python -m tests.test_augmentation
"""

import unittest
import torch
import numpy as np
from omegaconf import OmegaConf
import sys
import os
from monai.utils import set_determinism

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.augmentation import AugmentationPipeline2D
from src.utils.run_name import generate_augmentation_string


class TestTransformBuilding(unittest.TestCase):
    """Test that transforms are correctly built from configs."""
    
    def test_none_config_no_transforms(self):
        """none.yaml config should result in no transforms."""
        cfg = OmegaConf.load('configs/augmentation/none.yaml')
        pipeline = AugmentationPipeline2D(cfg)
        
        self.assertIsNone(pipeline.transform)
    
    def test_light_config_builds_transforms(self):
        """light_2d.yaml config should build flip + intensity transforms."""
        cfg = OmegaConf.load('configs/augmentation/light_2d.yaml')
        pipeline = AugmentationPipeline2D(cfg)
        
        self.assertIsNotNone(pipeline.transform)
        # Note: Depending on MONAI version/fallback, we check what we can
        if hasattr(pipeline.transform, 'transforms'):
            self.assertGreaterEqual(len(pipeline.transform.transforms), 3)
    
    def test_aggressive_config_builds_more_transforms(self):
        """aggressive_2d.yaml config should build more transforms than light."""
        cfg_light = OmegaConf.load('configs/augmentation/light_2d.yaml')
        cfg_agg = OmegaConf.load('configs/augmentation/aggressive_2d.yaml')
        
        pipeline_light = AugmentationPipeline2D(cfg_light)
        pipeline_agg = AugmentationPipeline2D(cfg_agg)
        
        if hasattr(pipeline_light.transform, 'transforms') and hasattr(pipeline_agg.transform, 'transforms'):
            num_light = len(pipeline_light.transform.transforms)
            num_agg = len(pipeline_agg.transform.transforms)
            self.assertGreater(num_agg, num_light)


class TestAugmentationApplication(unittest.TestCase):
    """Test that augmentations are correctly applied to data."""
    
    def test_none_config_preserves_data(self):
        """none config should return data unchanged."""
        cfg = OmegaConf.create({
            'spatial': {'enabled': False},
            'intensity': {'enabled': False}
        })
        pipeline = AugmentationPipeline2D(cfg)
        
        data = {
            'image': torch.randn(2, 128, 128),
            'label': torch.randint(0, 2, (1, 128, 128)).float()
        }
        out = pipeline(data)
        
        self.assertTrue(torch.allclose(out['image'], data['image']))
        self.assertTrue(torch.allclose(out['label'], data['label']))
    
    def test_flip_modifies_image(self):
        """Flip with prob=1.0 should modify image."""
        cfg = OmegaConf.create({
            'spatial': {
                'enabled': True,
                'random_flip': {
                    'enabled': True,
                    'prob': 1.0,
                    'spatial_axis': [0]
                }
            },
            'intensity': {'enabled': False}
        })
        pipeline = AugmentationPipeline2D(cfg)
        
        torch.manual_seed(42)
        data = {
            'image': torch.randn(2, 128, 128),
            'label': torch.randint(0, 2, (1, 128, 128)).float()
        }
        out = pipeline(data)
        
        # With prob=1.0, should definitely modify
        self.assertFalse(torch.allclose(out['image'], data['image']))
    
    def test_flip_preserves_shape(self):
        """Augmentation should preserve tensor shapes."""
        cfg = OmegaConf.load('configs/augmentation/light_2d.yaml')
        pipeline = AugmentationPipeline2D(cfg)
        
        data = {
            'image': torch.randn(2, 128, 128),
            'label': torch.randint(0, 2, (1, 128, 128)).float()
        }
        out = pipeline(data)
        
        self.assertEqual(out['image'].shape, data['image'].shape)
        self.assertEqual(out['label'].shape, data['label'].shape)
    
    def test_label_remains_binary(self):
        """Label should remain binary after spatial augmentation."""
        cfg = OmegaConf.create({
            'spatial': {
                'enabled': True,
                'random_flip': {
                    'enabled': True,
                    'prob': 1.0,
                    'spatial_axis': [0, 1]
                }
            },
            'intensity': {'enabled': False}
        })
        pipeline = AugmentationPipeline2D(cfg)
        
        data = {
            'image': torch.randn(2, 128, 128),
            'label': torch.randint(0, 2, (1, 128, 128)).float()
        }
        out = pipeline(data)
        
        # Label should still be binary (0 or 1)
        unique_vals = torch.unique(out['label'])
        self.assertLessEqual(len(unique_vals), 2)
        for v in unique_vals.tolist():
            self.assertIn(v, [0.0, 1.0])


class TestDeterministicBehavior(unittest.TestCase):
    """Test that augmentations are reproducible with same seed."""
    
    def test_same_seed_same_output(self):
        """Same seed should produce same augmentation."""
        cfg = OmegaConf.create({
            'spatial': {
                'enabled': True,
                'random_flip': {
                    'enabled': True,
                    'prob': 0.5,
                    'spatial_axis': [0, 1]
                }
            },
            'intensity': {
                'enabled': True,
                'random_shift': {
                    'enabled': True,
                    'prob': 0.5,
                    'offsets': 0.1
                },
                'random_scale': {
                    'enabled': True,
                    'prob': 0.5,
                    'factors': 0.1
                }
            }
        })
        
        data = {
            'image': torch.randn(2, 128, 128),
            'label': torch.randint(0, 2, (1, 128, 128)).float()
        }
        
        # First run with seed 42
        torch.manual_seed(42)
        np.random.seed(42)
        set_determinism(seed=42)
        pipeline1 = AugmentationPipeline2D(cfg)
        if hasattr(pipeline1.transform, 'set_random_state'):
            pipeline1.transform.set_random_state(seed=42)
        out1 = pipeline1(data.copy())
        
        # Second run with seed 42
        torch.manual_seed(42)
        np.random.seed(42)
        set_determinism(seed=42)
        pipeline2 = AugmentationPipeline2D(cfg)
        if hasattr(pipeline2.transform, 'set_random_state'):
            pipeline2.transform.set_random_state(seed=42)
        out2 = pipeline2(data.copy())
        
        # Should produce identical outputs
        self.assertTrue(torch.allclose(out1['image'], out2['image']))
        self.assertTrue(torch.allclose(out1['label'], out2['label']))
    
    def test_different_seed_different_output(self):
        """Different seeds should produce different augmentations."""
        cfg = OmegaConf.create({
            'spatial': {
                'enabled': True,
                'random_flip': {
                    'enabled': True,
                    'prob': 0.5,
                    'spatial_axis': [0, 1]
                }
            },
            'intensity': {'enabled': False}
        })
        
        data = {
            'image': torch.randn(2, 128, 128),
            'label': torch.randint(0, 2, (1, 128, 128)).float()
        }
        
        # Run with seed 42
        torch.manual_seed(42)
        np.random.seed(42)
        set_determinism(seed=42)
        pipeline1 = AugmentationPipeline2D(cfg)
        if hasattr(pipeline1.transform, 'set_random_state'):
            pipeline1.transform.set_random_state(seed=42)
        out1 = pipeline1(data.copy())
        
        # Run with seed 123
        torch.manual_seed(123)
        np.random.seed(123)
        set_determinism(seed=123)
        pipeline2 = AugmentationPipeline2D(cfg)
        if hasattr(pipeline2.transform, 'set_random_state'):
            pipeline2.transform.set_random_state(seed=123)
        out2 = pipeline2(data.copy())
        
        # Should produce different outputs (with high probability)
        self.assertFalse(torch.allclose(out1['image'], out2['image']))


class TestRunNameGeneration(unittest.TestCase):
    """Test that augmentation strategies are correctly encoded in run names."""
    
    def test_none_encoded_as_augNONE(self):
        """none.yaml should encode as augNONE."""
        cfg = OmegaConf.create({
            'spatial': {'enabled': False},
            'intensity': {'enabled': False},
            '_name_': 'none'
        })
        
        aug_str = generate_augmentation_string(cfg)
        self.assertEqual(aug_str, "augNONE")
    
    def test_none_config_none_encoded_as_augNONE(self):
        """None config should encode as augNONE."""
        aug_str = generate_augmentation_string(None)
        self.assertEqual(aug_str, "augNONE")
    
    def test_light_encoded_as_augLIGHT2D(self):
        """light_2d.yaml should encode as augLIGHT2D."""
        cfg = OmegaConf.create({
            'spatial': {'enabled': True},
            'intensity': {'enabled': True},
            '_name_': 'light_2d'
        })
        
        aug_str = generate_augmentation_string(cfg)
        self.assertEqual(aug_str, "augLIGHT2D")
    
    def test_aggressive_encoded_as_augAGG2D(self):
        """aggressive_2d.yaml should encode as augAGG2D."""
        cfg = OmegaConf.create({
            'spatial': {'enabled': True},
            'intensity': {'enabled': True},
            '_name_': 'aggressive_2d'
        })
        
        aug_str = generate_augmentation_string(cfg)
        self.assertEqual(aug_str, "augAGG2D")
    
    def test_custom_encoded_as_augCUSTOM(self):
        """Custom config should encode as augCUSTOM."""
        cfg = OmegaConf.create({
            'spatial': {'enabled': True},
            'intensity': {'enabled': True},
            '_name_': 'my_custom_config'
        })
        
        aug_str = generate_augmentation_string(cfg)
        self.assertEqual(aug_str, "augCUSTOM")


if __name__ == '__main__':
    unittest.main()
