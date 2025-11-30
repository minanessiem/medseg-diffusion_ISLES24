"""
Unit tests for Phase 1 checkpoint state functionality.

Tests:
- find_latest_checkpoint_step(): Finds correct step from checkpoint files
- save_interval_checkpoint(): Saves all three files with correct structure
- load_checkpoint(): Loads files and handles missing files gracefully

Run with:
    python -m unittest tests.test_checkpoint_state -v
    
Or from project root:
    python tests/test_checkpoint_state.py
"""

import os
import sys
import tempfile
import shutil
import unittest
import torch
import torch.nn as nn
from types import SimpleNamespace
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.checkpoint_utils import (
    find_latest_checkpoint_step,
    save_interval_checkpoint,
    load_checkpoint,
)


class MockModel(nn.Module):
    """Simple mock model for testing checkpoint save/load."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        self.conv = nn.Conv2d(1, 1, 3)
    
    def forward(self, x):
        return x


class MockScheduler:
    """Mock scheduler with state_dict support."""
    def __init__(self):
        self.last_epoch = 0
        self.base_lrs = [0.001]
    
    def state_dict(self):
        return {'last_epoch': self.last_epoch, 'base_lrs': self.base_lrs}
    
    def load_state_dict(self, state_dict):
        self.last_epoch = state_dict['last_epoch']
        self.base_lrs = state_dict['base_lrs']


def make_test_config():
    """Create a test config matching the training config structure."""
    return OmegaConf.create({
        'training': {
            'checkpoint_interval': {
                'enabled': True,
                'save_interval': 10,
                'keep_last_n': 3,
                'model_template': 'models/checkpoint/diffusion_chkpt_step_{:06d}.pth',
                'opt_template': 'models/checkpoint/opt_chkpt_step_{:06d}.pth',
                'state_template': 'models/checkpoint/training_state_step_{:06d}.pth',
            },
            'ema_rate_precision': 4,
        }
    })


class TestFindLatestCheckpointStep(unittest.TestCase):
    """Tests for find_latest_checkpoint_step()."""
    
    def setUp(self):
        """Create temp directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.temp_dir, 'models', 'checkpoint')
        os.makedirs(self.checkpoint_dir)
    
    def tearDown(self):
        """Clean up temp directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_finds_latest_step(self):
        """Should return highest step number from checkpoint files."""
        template = 'diffusion_chkpt_step_{:06d}.pth'
        
        # Create mock checkpoint files
        for step in [1000, 5000, 3000, 10000, 7000]:
            fpath = os.path.join(self.checkpoint_dir, template.format(step))
            torch.save({}, fpath)
        
        result = find_latest_checkpoint_step(self.checkpoint_dir, template)
        self.assertEqual(result, 10000)
    
    def test_returns_none_for_empty_dir(self):
        """Should return None if no matching checkpoints exist."""
        template = 'diffusion_chkpt_step_{:06d}.pth'
        result = find_latest_checkpoint_step(self.checkpoint_dir, template)
        self.assertIsNone(result)
    
    def test_returns_none_for_nonexistent_dir(self):
        """Should return None if directory doesn't exist."""
        template = 'diffusion_chkpt_step_{:06d}.pth'
        result = find_latest_checkpoint_step('/nonexistent/path', template)
        self.assertIsNone(result)
    
    def test_ignores_non_matching_files(self):
        """Should ignore files that don't match the template pattern."""
        template = 'diffusion_chkpt_step_{:06d}.pth'
        
        # Create matching file
        torch.save({}, os.path.join(self.checkpoint_dir, template.format(5000)))
        
        # Create non-matching files
        torch.save({}, os.path.join(self.checkpoint_dir, 'other_file.pth'))
        torch.save({}, os.path.join(self.checkpoint_dir, 'best_model_step_10000.pth'))
        
        result = find_latest_checkpoint_step(self.checkpoint_dir, template)
        self.assertEqual(result, 5000)


class TestSaveIntervalCheckpoint(unittest.TestCase):
    """Tests for save_interval_checkpoint()."""
    
    def setUp(self):
        """Create temp directory and mock objects."""
        self.temp_dir = tempfile.mkdtemp()
        self.run_dir = self.temp_dir + '/'  # Trailing slash to match real usage
        self.cfg = make_test_config()
        self.model = MockModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = MockScheduler()
        self.scheduler.last_epoch = 50
    
    def tearDown(self):
        """Clean up temp directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_saves_all_three_files(self):
        """Should save model, optimizer, and training state files."""
        step = 10000
        
        saved_files = save_interval_checkpoint(
            self.model, self.optimizer, step, self.cfg, self.run_dir,
            ema_params=[],
            ema_rates=[],
            scheduler=self.scheduler,
            best_metric_value=0.85,
            best_metric_step=8000,
        )
        
        # Check three files saved
        self.assertEqual(len(saved_files), 3)
        
        # Check files exist
        for fpath in saved_files:
            self.assertTrue(os.path.exists(fpath), f"File not found: {fpath}")
        
        # Check file names
        model_path = os.path.join(self.run_dir, 'models/checkpoint/diffusion_chkpt_step_010000.pth')
        opt_path = os.path.join(self.run_dir, 'models/checkpoint/opt_chkpt_step_010000.pth')
        state_path = os.path.join(self.run_dir, 'models/checkpoint/training_state_step_010000.pth')
        
        self.assertTrue(os.path.exists(model_path))
        self.assertTrue(os.path.exists(opt_path))
        self.assertTrue(os.path.exists(state_path))
    
    def test_training_state_contains_required_keys(self):
        """Training state should contain all required keys."""
        step = 5000
        ema_rates = [0.999, 0.9999]
        ema_params = [
            [p.clone() for p in self.model.parameters()],
            [p.clone() for p in self.model.parameters()],
        ]
        
        save_interval_checkpoint(
            self.model, self.optimizer, step, self.cfg, self.run_dir,
            ema_params=ema_params,
            ema_rates=ema_rates,
            scheduler=self.scheduler,
            best_metric_value=0.82,
            best_metric_step=4000,
        )
        
        # Load and check training state
        state_path = os.path.join(self.run_dir, 'models/checkpoint/training_state_step_005000.pth')
        state = torch.load(state_path)
        
        # Check required keys
        self.assertIn('global_step', state)
        self.assertIn('ema_rates', state)
        self.assertIn('ema_params', state)
        self.assertIn('scheduler_state_dict', state)
        self.assertIn('best_metric_value', state)
        self.assertIn('best_metric_step', state)
        
        # Check values
        self.assertEqual(state['global_step'], 5000)
        self.assertEqual(state['ema_rates'], [0.999, 0.9999])
        self.assertEqual(len(state['ema_params']), 2)  # Two EMA rates
        self.assertEqual(state['best_metric_value'], 0.82)
        self.assertEqual(state['best_metric_step'], 4000)
        self.assertEqual(state['scheduler_state_dict']['last_epoch'], 50)
    
    def test_handles_no_scheduler(self):
        """Should handle case where scheduler is None."""
        step = 1000
        
        saved_files = save_interval_checkpoint(
            self.model, self.optimizer, step, self.cfg, self.run_dir,
            ema_params=[],
            ema_rates=[],
            scheduler=None,  # No scheduler
            best_metric_value=None,
            best_metric_step=None,
        )
        
        # Should still save 3 files
        self.assertEqual(len(saved_files), 3)
        
        # Check scheduler_state_dict is None
        state_path = os.path.join(self.run_dir, 'models/checkpoint/training_state_step_001000.pth')
        state = torch.load(state_path)
        self.assertIsNone(state['scheduler_state_dict'])
    
    def test_ema_params_on_cpu(self):
        """EMA params should be saved as CPU tensors."""
        step = 2000
        ema_params = [[p.clone() for p in self.model.parameters()]]
        
        save_interval_checkpoint(
            self.model, self.optimizer, step, self.cfg, self.run_dir,
            ema_params=ema_params,
            ema_rates=[0.9999],
            scheduler=None,
        )
        
        state_path = os.path.join(self.run_dir, 'models/checkpoint/training_state_step_002000.pth')
        state = torch.load(state_path)
        
        # Check all EMA params are on CPU
        for rate_params in state['ema_params']:
            for param in rate_params:
                self.assertEqual(param.device.type, 'cpu')


class TestLoadCheckpoint(unittest.TestCase):
    """Tests for load_checkpoint()."""
    
    def setUp(self):
        """Create temp directory with checkpoint files."""
        self.temp_dir = tempfile.mkdtemp()
        self.run_dir = self.temp_dir + '/'
        self.cfg = make_test_config()
        self.device = torch.device('cpu')
        
        # Create checkpoint directory
        self.checkpoint_dir = os.path.join(self.temp_dir, 'models', 'checkpoint')
        os.makedirs(self.checkpoint_dir)
        
        # Create mock model and save checkpoints
        self.model = MockModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    
    def tearDown(self):
        """Clean up temp directory."""
        shutil.rmtree(self.temp_dir)
    
    def _save_mock_checkpoint(self, step):
        """Helper to save a complete mock checkpoint at given step."""
        save_interval_checkpoint(
            self.model, self.optimizer, step, self.cfg, self.run_dir,
            ema_params=[[p.clone() for p in self.model.parameters()]],
            ema_rates=[0.9999],
            scheduler=MockScheduler(),
            best_metric_value=0.85,
            best_metric_step=step - 1000,
        )
    
    def test_loads_complete_checkpoint(self):
        """Should load all checkpoint components."""
        self._save_mock_checkpoint(10000)
        
        result = load_checkpoint(self.run_dir, '10000', self.cfg, self.device)
        
        # Check returned structure
        self.assertIn('model_state_dict', result)
        self.assertIn('optimizer_state_dict', result)
        self.assertIn('training_state', result)
        self.assertIn('global_step', result)
        
        # Check values
        self.assertEqual(result['global_step'], 10000)
        self.assertIsNotNone(result['model_state_dict'])
        self.assertIsNotNone(result['optimizer_state_dict'])
        self.assertEqual(result['training_state']['best_metric_value'], 0.85)
    
    def test_loads_latest_checkpoint(self):
        """Should find and load the latest checkpoint when step='latest'."""
        # Save multiple checkpoints
        self._save_mock_checkpoint(5000)
        self._save_mock_checkpoint(15000)
        self._save_mock_checkpoint(10000)
        
        result = load_checkpoint(self.run_dir, 'latest', self.cfg, self.device)
        
        # Should load step 15000 (highest)
        self.assertEqual(result['global_step'], 15000)
    
    def test_handles_missing_training_state(self):
        """Should gracefully handle missing training_state file (backwards compat)."""
        step = 8000
        
        # Save only model and optimizer (simulate old checkpoint)
        model_path = os.path.join(self.checkpoint_dir, f'diffusion_chkpt_step_{step:06d}.pth')
        opt_path = os.path.join(self.checkpoint_dir, f'opt_chkpt_step_{step:06d}.pth')
        
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), opt_path)
        
        result = load_checkpoint(self.run_dir, '8000', self.cfg, self.device)
        
        # Should still work with default training_state
        self.assertEqual(result['global_step'], 8000)
        self.assertIsNotNone(result['model_state_dict'])
        self.assertIsNotNone(result['optimizer_state_dict'])
        # training_state should have minimal defaults
        self.assertEqual(result['training_state']['global_step'], 8000)
    
    def test_handles_missing_optimizer(self):
        """Should handle missing optimizer file gracefully."""
        step = 6000
        
        # Save only model (no optimizer)
        model_path = os.path.join(self.checkpoint_dir, f'diffusion_chkpt_step_{step:06d}.pth')
        torch.save(self.model.state_dict(), model_path)
        
        result = load_checkpoint(self.run_dir, '6000', self.cfg, self.device)
        
        # optimizer_state_dict should be None
        self.assertIsNone(result['optimizer_state_dict'])
        self.assertIsNotNone(result['model_state_dict'])
    
    def test_raises_on_missing_model(self):
        """Should raise FileNotFoundError if model checkpoint missing."""
        with self.assertRaises(FileNotFoundError):
            load_checkpoint(self.run_dir, '99999', self.cfg, self.device)
    
    def test_raises_on_no_checkpoints_for_latest(self):
        """Should raise FileNotFoundError if no checkpoints exist for 'latest'."""
        with self.assertRaises(FileNotFoundError):
            load_checkpoint(self.run_dir, 'latest', self.cfg, self.device)


class TestSaveLoadRoundTrip(unittest.TestCase):
    """Integration tests for save → load round trip."""
    
    def setUp(self):
        """Create temp directory and mock objects."""
        self.temp_dir = tempfile.mkdtemp()
        self.run_dir = self.temp_dir + '/'
        self.cfg = make_test_config()
        self.device = torch.device('cpu')
        
        # Create models
        self.model = MockModel()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = MockScheduler()
        self.scheduler.last_epoch = 100
    
    def tearDown(self):
        """Clean up temp directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_round_trip_preserves_state(self):
        """Save then load should preserve all state."""
        step = 25000
        ema_rates = [0.999, 0.9999]
        ema_params = [
            [p.clone() for p in self.model.parameters()],
            [p.clone() for p in self.model.parameters()],
        ]
        best_metric_value = 0.8765
        best_metric_step = 24000
        
        # Modify model weights for verification
        with torch.no_grad():
            self.model.linear.weight.fill_(0.123)
        
        # Take optimizer step to get non-trivial state
        self.optimizer.zero_grad()
        dummy_loss = self.model.linear.weight.sum()
        dummy_loss.backward()
        self.optimizer.step()
        
        # Save checkpoint
        save_interval_checkpoint(
            self.model, self.optimizer, step, self.cfg, self.run_dir,
            ema_params=ema_params,
            ema_rates=ema_rates,
            scheduler=self.scheduler,
            best_metric_value=best_metric_value,
            best_metric_step=best_metric_step,
        )
        
        # Load checkpoint
        result = load_checkpoint(self.run_dir, str(step), self.cfg, self.device)
        
        # Verify global step
        self.assertEqual(result['global_step'], step)
        
        # Verify model weights can be loaded
        new_model = MockModel()
        new_model.load_state_dict(result['model_state_dict'])
        self.assertTrue(torch.allclose(
            new_model.linear.weight, 
            self.model.linear.weight,
            atol=1e-6
        ))
        
        # Verify optimizer state can be loaded
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        new_optimizer.load_state_dict(result['optimizer_state_dict'])
        
        # Verify training state
        ts = result['training_state']
        self.assertEqual(ts['ema_rates'], ema_rates)
        self.assertEqual(len(ts['ema_params']), 2)
        self.assertEqual(ts['scheduler_state_dict']['last_epoch'], 100)
        self.assertEqual(ts['best_metric_value'], best_metric_value)
        self.assertEqual(ts['best_metric_step'], best_metric_step)


if __name__ == '__main__':
    # Run with verbosity
    unittest.main(verbosity=2)

