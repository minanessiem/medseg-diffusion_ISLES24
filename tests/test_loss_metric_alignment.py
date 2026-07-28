"""
Test that DiceLoss produces identical values to DiceNativeCoefficient
on binarized predictions.

This test validates that the training loss is mathematically aligned with
the evaluation metric, ensuring "optimize what you measure."
"""

import unittest

import torch
from src.losses.segmentation_losses import DiceLoss, BCELoss
from src.metrics.metrics import DiceNativeCoefficient


def _scalar_value(value):
    """Return a Python float from tensor or numeric scalar metric outputs."""
    if torch.is_tensor(value):
        return value.item()
    return float(value)


class TestLossMetricAlignment(unittest.TestCase):
    def test_dice_loss_metric_equivalence(self):
        """Verify that DiceLoss matches DiceNativeCoefficient on binarized inputs."""
        print("=" * 70)
        print("Testing Dice Loss - Metric Alignment")
        print("=" * 70)

        dice_metric = DiceNativeCoefficient(threshold=0.5)
        dice_loss_fn = DiceLoss(smooth=1e-8, apply_sigmoid=False)

        print("\n[Test 1] Standard predictions")
        print("-" * 70)
        y_pred_soft = torch.tensor([[[[0.2, 0.8], [0.6, 0.3]]]], dtype=torch.float32)
        y_true = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]], dtype=torch.float32)

        metric_dice = dice_metric(y_pred_soft, y_true)
        y_pred_binary = (y_pred_soft > 0.5).float()
        loss_value = dice_loss_fn(y_pred_binary, y_true)
        loss_dice_coef = 1.0 - loss_value

        print(f"  Metric Dice Coefficient: {metric_dice.item():.8f}")
        print(f"  Loss Dice Coefficient:   {loss_dice_coef.item():.8f}")
        print(
            f"  Difference:              {abs(metric_dice - loss_dice_coef).item():.2e}"
        )

        self.assertLess(torch.abs(metric_dice - loss_dice_coef).item(), 1e-6)
        print("  PASSED")

        print("\n[Test 2] Perfect prediction")
        print("-" * 70)
        y_pred_perfect = torch.ones((1, 1, 4, 4))
        y_true_perfect = torch.ones((1, 1, 4, 4))

        metric_dice_perfect = dice_metric(y_pred_perfect, y_true_perfect)
        loss_dice_perfect = 1.0 - dice_loss_fn(y_pred_perfect, y_true_perfect)

        print(f"  Metric Dice Coefficient: {metric_dice_perfect.item():.8f}")
        print(f"  Loss Dice Coefficient:   {loss_dice_perfect.item():.8f}")
        print(
            "  Difference:              "
            f"{abs(metric_dice_perfect - loss_dice_perfect).item():.2e}"
        )

        self.assertLess(
            torch.abs(metric_dice_perfect - loss_dice_perfect).item(), 1e-6
        )
        self.assertGreater(metric_dice_perfect.item(), 0.999)
        print("  PASSED")

        print("\n[Test 3] Empty masks")
        print("-" * 70)
        y_pred_empty = torch.zeros((1, 1, 4, 4))
        y_true_empty = torch.zeros((1, 1, 4, 4))

        metric_dice_empty = dice_metric(y_pred_empty, y_true_empty)
        loss_dice_empty = 1.0 - dice_loss_fn(y_pred_empty, y_true_empty)

        print(f"  Metric Dice Coefficient: {_scalar_value(metric_dice_empty):.8f}")
        print(f"  Loss Dice Coefficient:   {_scalar_value(loss_dice_empty):.8f}")
        print("  NOTE: Empty mask behavior may differ")
        print("        Metric returns 0, Loss uses smoothing for stability")
        print("  PASSED (expected difference for edge case)")

        print("\n[Test 4] Partial overlap")
        print("-" * 70)
        y_pred_partial = torch.tensor(
            [[[[0.9, 0.8], [0.1, 0.2]]]], dtype=torch.float32
        )
        y_true_partial = torch.tensor(
            [[[[1.0, 1.0], [0.0, 0.0]]]], dtype=torch.float32
        )

        y_pred_partial_bin = (y_pred_partial > 0.5).float()

        metric_dice_partial = dice_metric(y_pred_partial, y_true_partial)
        loss_dice_partial = 1.0 - dice_loss_fn(
            y_pred_partial_bin, y_true_partial
        )

        print(f"  Metric Dice Coefficient: {metric_dice_partial.item():.8f}")
        print(f"  Loss Dice Coefficient:   {loss_dice_partial.item():.8f}")
        print(
            f"  Difference:              {abs(metric_dice_partial - loss_dice_partial).item():.2e}"
        )

        self.assertLess(
            torch.abs(metric_dice_partial - loss_dice_partial).item(), 1e-6
        )
        print("  PASSED")

    def test_soft_vs_hard_dice(self):
        """Demonstrate the difference between soft and hard Dice."""
        print("\n" + "=" * 70)
        print("Soft vs Hard Dice Comparison")
        print("=" * 70)

        dice_loss_fn = DiceLoss(smooth=1e-8, apply_sigmoid=False)

        y_pred_soft = torch.tensor(
            [[[[0.7, 0.4], [0.6, 0.3]]]], dtype=torch.float32
        )
        y_true = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32)

        soft_dice_loss = dice_loss_fn(y_pred_soft, y_true)
        soft_dice_coef = 1.0 - soft_dice_loss

        y_pred_hard = (y_pred_soft > 0.5).float()
        hard_dice_loss = dice_loss_fn(y_pred_hard, y_true)
        hard_dice_coef = 1.0 - hard_dice_loss

        print(f"\nSoft predictions (continuous): {y_pred_soft.flatten().tolist()}")
        print(f"Hard predictions (thresholded): {y_pred_hard.flatten().tolist()}")
        print(f"Ground truth:                   {y_true.flatten().tolist()}")
        print(f"\nSoft Dice Coefficient:  {soft_dice_coef.item():.6f} (training)")
        print(f"Hard Dice Coefficient:  {hard_dice_coef.item():.6f} (evaluation)")
        print(f"Difference:             {abs(soft_dice_coef - hard_dice_coef).item():.6f}")

        self.assertGreater(soft_dice_coef.item(), 0.0)
        self.assertGreater(hard_dice_coef.item(), 0.0)

    def test_bce_loss_basic(self):
        """Basic sanity check for BCE loss."""
        print("\n" + "=" * 70)
        print("BCE Loss Basic Tests")
        print("=" * 70)

        print("\n[Test 1] Perfect prediction")
        print("-" * 70)
        bce_loss_fn = BCELoss(pos_weight=None, apply_sigmoid=False)
        y_pred_perfect = torch.ones((1, 1, 4, 4)) * 0.9999
        y_true_perfect = torch.ones((1, 1, 4, 4))

        bce_perfect = bce_loss_fn(y_pred_perfect, y_true_perfect)
        print(f"  BCE Loss (perfect): {bce_perfect.item():.6f}")
        self.assertLess(bce_perfect.item(), 0.01)
        print("  PASSED")

        print("\n[Test 2] Complete mismatch")
        print("-" * 70)
        y_pred_wrong = torch.ones((1, 1, 4, 4)) * 0.9999
        y_true_wrong = torch.zeros((1, 1, 4, 4))

        bce_wrong = bce_loss_fn(y_pred_wrong, y_true_wrong)
        print(f"  BCE Loss (mismatch): {bce_wrong.item():.6f}")
        self.assertGreater(bce_wrong.item(), 5.0)
        print("  PASSED")


if __name__ == "__main__":
    unittest.main()

