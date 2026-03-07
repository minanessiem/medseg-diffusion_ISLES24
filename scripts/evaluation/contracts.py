"""
Shared contracts for the greenfield evaluation package.

These dataclasses are intentionally lightweight and focused on:
- sample payload boundaries between IO adapters and metric engine
- protocol settings for threshold evaluation
- explicit reporting scopes and running-stat semantics
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from torch import Tensor

ScopeName = Literal["all_slices", "foreground_only"]


@dataclass
class SliceSample:
    """
    One evaluation sample yielded by a streaming producer.

    Exactly one of `prediction_prob` or `prediction_mask` should be provided.
    """

    case_id: str
    slice_id: str
    ground_truth_mask: Tensor
    prediction_prob: Optional[Tensor] = None
    prediction_mask: Optional[Tensor] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate basic sample invariants."""
        if self.prediction_prob is None and self.prediction_mask is None:
            raise ValueError(
                "SliceSample must provide either prediction_prob or prediction_mask."
            )
        if self.prediction_prob is not None and self.prediction_mask is not None:
            raise ValueError(
                "SliceSample must not provide both prediction_prob and prediction_mask."
            )


@dataclass(frozen=True)
class ThresholdProtocol:
    """Protocol settings used by evaluation orchestration."""

    mode: Literal["fixed", "sweep"]
    thresholds: List[float]
    optimize_metric: Optional[str] = None


@dataclass
class RunningStats:
    """
    Numerically stable-enough running statistics for mean/std.

    Tracks values using sum and sum-of-squares to avoid storing samples.
    """

    count: int = 0
    value_sum: float = 0.0
    value_sum_sq: float = 0.0

    def update(self, value: float) -> None:
        """Update with one scalar value."""
        self.count += 1
        self.value_sum += value
        self.value_sum_sq += value * value

    @property
    def mean(self) -> float:
        """Return running mean."""
        if self.count == 0:
            return 0.0
        return self.value_sum / self.count

    @property
    def std(self) -> float:
        """Return population standard deviation."""
        if self.count == 0:
            return 0.0
        mean = self.mean
        variance = (self.value_sum_sq / self.count) - (mean * mean)
        if variance < 0:
            variance = 0.0
        return variance ** 0.5

    def to_dict(self) -> Dict[str, float]:
        """Serialize to a mean/std/count dictionary."""
        return {
            "mean": float(self.mean),
            "std": float(self.std),
            "count": int(self.count),
        }


@dataclass
class ScopedRunningStats:
    """Holds running stats for both denominator scopes."""

    all_slices: RunningStats = field(default_factory=RunningStats)
    foreground_only: RunningStats = field(default_factory=RunningStats)

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """Serialize to scope-keyed dictionary."""
        return {
            "all_slices": self.all_slices.to_dict(),
            "foreground_only": self.foreground_only.to_dict(),
        }

