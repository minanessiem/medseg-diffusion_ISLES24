# Evaluation Package

This package contains the greenfield, unified segmentation evaluation stack.

Core design principles:
- Keep source-specific loading/inference separate from metric computation.
- Reuse existing metric implementations from `src/metrics/metrics.py`.
- Stream samples out-of-core (no full-dataset buffering in memory).
- Emit a canonical reporting schema across entrypoints.

Implemented so far (Phase 1 through Phase 3):
- `contracts.py`: shared typed structures between IO, engine, and reporting.
- `metrics_registry.py`: threshold-aware wrappers copied/adapted from analysis.
- `metrics_engine.py`: streaming accumulators for threshold/scoped metric summaries.
- `threshold_protocol.py`: fixed/sweep threshold policy helpers.
- `mask_builder.py`: probability/mask normalization and binary conversion utilities.
- `io_nnunet.py`: streaming producer for post-threshold nnU-Net masks.
- `io_diffusion.py`: streaming producer for probability predictions from model inference.
- `reporting.py`: canonical JSON/CSV/text reporting helpers.
- `compute_segmentation_metrics_for_nnunet_2d_predictions.py`: nnU-Net v2 entrypoint.
- `compute_segmentation_metrics_for_diffusionmodel_2d_predictions.py`: custom-model v2 entrypoint.
- `slurm_runners/`: SLURM submission wrappers for both v2 entrypoints.

Useful dev/testing options:
- diffusion entrypoint supports `--test --test-max-slices 10` for fast iteration.
- diffusion entrypoint supports multi-case ensembles via `--ensemble-samples 1,3,5`
  and `--ensemble-method both`, producing case-prefixed report files.
- canonical JSON includes `metrics.default_threshold_metrics` for threshold=0.5 when evaluated.

