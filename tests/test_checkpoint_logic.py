"""Unit tests for checkpoint decision logic."""

import unittest
import math
from types import SimpleNamespace
from src.training.checkpoint_utils import should_save_best_checkpoint


class TestShouldSaveBestCheckpoint(unittest.TestCase):
    """Tests for should_save_best_checkpoint function."""
    
    def test_first_validation_max_mode(self):
        """First validation should always save (max mode)."""
        val_results = {'dice_2d_fg': 0.75}
        config = SimpleNamespace(metric_name='dice_2d_fg', metric_mode='max')
        
        should_save, reason, new_best = should_save_best_checkpoint(
            val_results, float('-inf'), config
        )
        
        self.assertTrue(should_save)
        self.assertIn('first best model', reason.lower())
        self.assertIn('baseline', reason.lower())
        self.assertEqual(new_best, 0.75)
    
    def test_first_validation_min_mode(self):
        """First validation should always save (min mode)."""
        val_results = {'loss': 0.25}
        config = SimpleNamespace(metric_name='loss', metric_mode='min')
        
        should_save, reason, new_best = should_save_best_checkpoint(
            val_results, float('inf'), config
        )
        
        self.assertTrue(should_save)
        self.assertIn('first best model', reason.lower())
        self.assertEqual(new_best, 0.25)
    
    def test_improvement_max_mode(self):
        """Should save when metric improves (increases) in max mode."""
        val_results = {'dice_2d_fg': 0.85}
        config = SimpleNamespace(metric_name='dice_2d_fg', metric_mode='max')
        
        should_save, reason, new_best = should_save_best_checkpoint(
            val_results, 0.80, config
        )
        
        self.assertTrue(should_save)
        self.assertIn('improved', reason.lower())
        self.assertIn('0.8000', reason)
        self.assertIn('0.8500', reason)
        self.assertEqual(new_best, 0.85)
    
    def test_no_improvement_max_mode(self):
        """Should not save when metric doesn't improve in max mode."""
        val_results = {'dice_2d_fg': 0.75}
        config = SimpleNamespace(metric_name='dice_2d_fg', metric_mode='max')
        
        should_save, reason, new_best = should_save_best_checkpoint(
            val_results, 0.80, config
        )
        
        self.assertFalse(should_save)
        self.assertIn('did not improve', reason.lower())
        self.assertEqual(new_best, 0.80)  # Unchanged
    
    def test_improvement_min_mode(self):
        """Should save when metric improves (decreases) in min mode."""
        val_results = {'loss': 0.15}
        config = SimpleNamespace(metric_name='loss', metric_mode='min')
        
        should_save, reason, new_best = should_save_best_checkpoint(
            val_results, 0.20, config
        )
        
        self.assertTrue(should_save)
        self.assertIn('improved', reason.lower())
        self.assertEqual(new_best, 0.15)
    
    def test_no_improvement_min_mode(self):
        """Should not save when metric doesn't improve in min mode."""
        val_results = {'loss': 0.25}
        config = SimpleNamespace(metric_name='loss', metric_mode='min')
        
        should_save, reason, new_best = should_save_best_checkpoint(
            val_results, 0.20, config
        )
        
        self.assertFalse(should_save)
        self.assertIn('did not improve', reason.lower())
        self.assertEqual(new_best, 0.20)  # Unchanged
    
    def test_nan_metric_value(self):
        """Should not save when metric is NaN."""
        val_results = {'dice_2d_fg': float('nan')}
        config = SimpleNamespace(metric_name='dice_2d_fg', metric_mode='max')
        
        should_save, reason, new_best = should_save_best_checkpoint(
            val_results, 0.80, config
        )
        
        self.assertFalse(should_save)
        self.assertIn('not finite', reason.lower())
        self.assertEqual(new_best, 0.80)  # Unchanged
    
    def test_inf_metric_value(self):
        """Should not save when metric is inf."""
        val_results = {'loss': float('inf')}
        config = SimpleNamespace(metric_name='loss', metric_mode='min')
        
        should_save, reason, new_best = should_save_best_checkpoint(
            val_results, 0.20, config
        )
        
        self.assertFalse(should_save)
        self.assertIn('not finite', reason.lower())
        self.assertEqual(new_best, 0.20)  # Unchanged
    
    def test_metric_not_found(self):
        """Should raise KeyError with helpful message when metric missing."""
        val_results = {'dice_2d_fg': 0.85, 'f1_2d': 0.82}
        config = SimpleNamespace(metric_name='iou_score', metric_mode='max')
        
        with self.assertRaises(KeyError) as context:
            should_save_best_checkpoint(val_results, 0.80, config)
        
        error_msg = str(context.exception)
        self.assertIn('iou_score', error_msg)
        self.assertIn('not found', error_msg)
        self.assertIn('dice_2d_fg', error_msg)
        self.assertIn('f1_2d', error_msg)
    
    def test_equal_values_max_mode(self):
        """Should not save when values are equal in max mode."""
        val_results = {'dice_2d_fg': 0.80}
        config = SimpleNamespace(metric_name='dice_2d_fg', metric_mode='max')
        
        should_save, reason, new_best = should_save_best_checkpoint(
            val_results, 0.80, config
        )
        
        self.assertFalse(should_save)
        self.assertEqual(new_best, 0.80)
    
    def test_equal_values_min_mode(self):
        """Should not save when values are equal in min mode."""
        val_results = {'loss': 0.20}
        config = SimpleNamespace(metric_name='loss', metric_mode='min')
        
        should_save, reason, new_best = should_save_best_checkpoint(
            val_results, 0.20, config
        )
        
        self.assertFalse(should_save)
        self.assertEqual(new_best, 0.20)


if __name__ == '__main__':
    unittest.main()

