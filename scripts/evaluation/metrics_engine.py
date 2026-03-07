"""
Streaming metrics engine for greenfield segmentation evaluation.

This module intentionally avoids storing per-sample tensors in memory.
It updates running statistics incrementally for each incoming sample.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from scripts.evaluation.contracts import ScopeName, ScopedRunningStats, SliceSample
from scripts.evaluation.metrics_registry import compute_metrics_at_threshold


DEFAULT_METRIC_NAMES: Sequence[str] = (
    "dice",
    "precision",
    "recall",
    "specificity",
    "f1",
    "f2",
)


@dataclass
class ThresholdState:
    """Accumulator state for one threshold."""

    threshold: float
    metrics: Dict[str, ScopedRunningStats] = field(default_factory=dict)
    slice_counts: Dict[str, int] = field(
        default_factory=lambda: {"total": 0, "foreground": 0, "empty": 0}
    )

    def to_dict(self) -> Dict[str, object]:
        """Serialize threshold state to report-ready dictionary."""
        return {
            "threshold": float(self.threshold),
            "slice_counts": {
                "total": int(self.slice_counts["total"]),
                "foreground": int(self.slice_counts["foreground"]),
                "empty": int(self.slice_counts["empty"]),
            },
            "metrics": {name: scoped.to_dict() for name, scoped in self.metrics.items()},
        }


class StreamingMetricsEngine:
    """
    Streaming evaluation engine over one or many thresholds.

    - Input is a stream of `SliceSample`
    - Output is running aggregated metrics with explicit denominator scopes
    """

    def __init__(
        self,
        thresholds: Sequence[float],
        metric_names: Optional[Sequence[str]] = None,
    ):
        if not thresholds:
            raise ValueError("StreamingMetricsEngine requires at least one threshold.")
        self.thresholds: List[float] = [float(t) for t in thresholds]
        self.metric_names: Sequence[str] = metric_names or DEFAULT_METRIC_NAMES
        self._states: Dict[float, ThresholdState] = {
            t: ThresholdState(
                threshold=t,
                metrics={name: ScopedRunningStats() for name in self.metric_names},
            )
            for t in self.thresholds
        }

    def update(self, sample: SliceSample) -> None:
        """Consume one sample and update all threshold accumulators."""
        sample.validate()
        gt = sample.ground_truth_mask
        has_foreground = bool((gt > 0.5).sum() > 0)
        pred_input = self._resolve_prediction_input(sample)

        for threshold in self.thresholds:
            state = self._states[threshold]
            state.slice_counts["total"] += 1
            if has_foreground:
                state.slice_counts["foreground"] += 1
            else:
                state.slice_counts["empty"] += 1

            metric_values = compute_metrics_at_threshold(pred_input, gt, threshold=threshold)
            self._update_metric_scopes(state, metric_values, has_foreground)

    def run(self, samples: Iterable[SliceSample]) -> Dict[float, Dict[str, object]]:
        """Consume a stream of samples and return finalized threshold results."""
        for sample in samples:
            self.update(sample)
        return self.finalize()

    def finalize(self) -> Dict[float, Dict[str, object]]:
        """Return threshold-keyed finalized metrics."""
        return {threshold: state.to_dict() for threshold, state in self._states.items()}

    @staticmethod
    def _resolve_prediction_input(sample: SliceSample):
        """
        Resolve the prediction tensor used by the metric wrappers.

        For probability sources, we pass probabilities and rely on thresholding
        in the wrapper.
        For post-threshold sources (e.g. nnU-Net masks), we pass the mask tensor.
        """
        if sample.prediction_prob is not None:
            return sample.prediction_prob
        if sample.prediction_mask is not None:
            return sample.prediction_mask
        raise ValueError("SliceSample has neither prediction_prob nor prediction_mask.")

    def _update_metric_scopes(
        self,
        state: ThresholdState,
        metric_values: Dict[str, float],
        has_foreground: bool,
    ) -> None:
        """Update all-slices and foreground-only running stats per metric."""
        for metric_name in self.metric_names:
            value = float(metric_values.get(metric_name, 0.0))
            scoped = state.metrics[metric_name]
            scoped.all_slices.update(value)
            if has_foreground:
                scoped.foreground_only.update(value)


def evaluate_stream(
    samples: Iterable[SliceSample],
    thresholds: Sequence[float],
    metric_names: Optional[Sequence[str]] = None,
) -> Dict[float, Dict[str, object]]:
    """Convenience function to evaluate a sample stream in one call."""
    engine = StreamingMetricsEngine(thresholds=thresholds, metric_names=metric_names)
    return engine.run(samples)


def get_scope_metric(
    finalized_results: Dict[float, Dict[str, object]],
    threshold: float,
    metric_name: str,
    scope: ScopeName,
) -> Dict[str, float]:
    """Utility accessor for tests and downstream report adapters."""
    return finalized_results[threshold]["metrics"][metric_name][scope]

