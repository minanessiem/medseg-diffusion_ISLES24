# Product Requirements Document: Config-Driven Custom Model Evaluation and Thresholding

**Document Version:** 1.0  
**Date:** 2026-06-11  
**Status:** Draft - Pending Review  
**Scope:** `scripts/evaluation` repository-model live inference evaluation, threshold sweep, and oracle threshold analysis

---

## 1. Overview and Task Description

### 1.1 Background

The repository contains multiple evaluation and analysis paths created at different stages of the project:

- `scripts/nnunet/` contains a newer config-driven nnU-Net evaluation path for file-based prediction/ground-truth evaluation.
- `scripts/evaluation/` contains reusable evaluation primitives, including threshold protocols, streaming metrics, 2D and 3D metric registries, reporting helpers, and legacy 2D live-model entrypoints.
- `scripts/analysis/threshold_analysis.py` is older and mostly 2D/slice-oriented. It overlaps with functionality now better suited to `scripts/evaluation`.

The current scientific need is threshold and calibration error-budget analysis for repository-trained models, especially ISLES26 3D discriminative segmentation models trained with random patches and validated with sliding-window inference over full volumes.

The current best DynUNet results suggest that blindly changing losses or model recipes is less valuable than first quantifying threshold sensitivity:

- How much does fixed threshold `0.5` cost relative to the best global threshold?
- How much ceiling remains if each case could choose its own oracle threshold?
- Are predicted volumes systematically biased relative to ground-truth volumes?

### 1.2 Problem Statement

The current repository-model evaluation path is not sufficiently config-driven, is still organized around historical 2D diffusion/discriminative slice evaluation, and does not provide first-class 3D volume threshold analysis with global and oracle threshold summaries.

The evaluation workflow should be able to:

1. Load a trained run's Hydra config and checkpoint.
2. Apply evaluation-time overrides using the same config philosophy as training.
3. Reuse existing repository dataloaders and validation inference logic.
4. Evaluate fixed thresholds, threshold sweeps, and per-case oracle thresholds.
5. Produce clear artifacts comparing threshold `0.5`, best global threshold, and oracle threshold ceilings.

### 1.3 Goal

Build a config-driven model evaluation pipeline inside `scripts/evaluation` that performs live inference from repository-trained models and supports configurable threshold protocols for both 2D slice and 3D volume model outputs.

The new pipeline should subsume the relevant threshold-analysis functionality currently living under `scripts/analysis`, while keeping the existing file-based nnU-Net evaluation path separate.

### 1.4 Non-Goals

- Do not merge nnU-Net file-based evaluation and repository-model live inference into a single monolithic pipeline.
- Do not create a separate ISLES26 volume reader.
- Do not duplicate preprocessing or volume loading logic outside `src.data.loaders`.
- Do not implement 3D diffusion volume evaluation in this change. The 3D IO module should be representation-based and future-compatible, but current non-discriminative diffusion adapters are 2D-shaped and should fail with an explicit unsupported-capability error.
- Do not remove `scripts/analysis/threshold_analysis.py` immediately. Deprecation can happen after feature parity and validation.

---

## 2. Requirements and Scope

### 2.1 Functional Requirements

#### FR-1: Config-Driven Runtime

- Load the baseline config from `<run_dir>/.hydra/config.yaml`.
- Apply evaluation-specific config composition and CLI overrides after loading the run config.
- Support overrides for:
  - data profile or data mode where valid
  - `dataset.active_subsets.val`, especially `val_fast` and `val_full`
  - `validation` config group, including `sliding_window_3d_metrics_subset` and `sliding_window_3d_metrics_full`
  - threshold protocol
  - output directory
  - device
  - checkpoint selection
- Preserve the default behavior of evaluating the model according to its original run config when no overrides are provided.
- Local evaluation scripts may use Hydra/OmegaConf directly. SLURM runner scripts under `scripts/evaluation/slurm_runners/` must not require OmegaConf on the cluster submission side; they should build command strings with the existing ersatz config tooling in `scripts/slurm/utils/commandline_utils.py`.

#### FR-2: Model Loading

- Accept a run directory and model checkpoint name.
- Locate checkpoints in established locations:
  - `<run_dir>/models/best/<model_name>.pth`
  - `<run_dir>/models/checkpoints/<model_name>.pth`
  - `<run_dir>/models/<model_name>.pth`
- Support EMA checkpoint lookup if currently supported by existing evaluation code.
- Rebuild the model using existing repository model factories and diffusion/discriminative adapters.
- Preserve the existing DDP/adapter prefix stripping behavior where needed.

#### FR-3: Data Loading and Validation Inference

- Use `src.data.loaders.get_dataloaders(cfg)` for all repository-model evaluation data access.
- Use existing repository preprocessing and loader contracts from:
  - `configs/data_profile/*`
  - `configs/dataset/*`
  - `configs/data_mode/*`
  - `configs/data_io/*`
- Use the validation dataloader returned under `dataloaders["val"]`.
- For `random_patches_3d`, rely on current repository behavior where training uses random patches but validation uses full volumes.
- Reuse existing validation inference policy, including sliding-window inference for 3D discriminative models.

#### FR-4: Supported Model Evaluation Modes

Support:

- 2D diffusion slices.
- 2D discriminative slices.
- 3D discriminative volumes.

Explicitly reject for the current implementation:

- 3D non-discriminative diffusion volumes, with a clear error message explaining that current diffusion adapters hard-code 2D sampling shapes and do not yet satisfy the 3D volume inference contract.

The IO boundary should be organized by representation rather than training technique:

- 2D live-model slice IO should support diffusion and discriminative models when their outputs can be normalized to probability slices.
- 3D live-model volume IO should accept any future model or adapter that returns probability volumes shaped like the ground-truth target. In the current repo, that means 3D discriminative adapters only.

#### FR-5: Threshold Protocols

Provide config-controlled threshold protocols:

- `fixed`: evaluate exactly one threshold, usually `0.5`.
- `sweep`: evaluate a configured threshold list/range and identify the best global threshold.
- `oracle_per_case`: choose the best threshold per case according to a configured primary metric.
- `sweep_with_oracle`: evaluate the global sweep and additionally compute the per-case oracle ceiling.

Default sweep thresholds should cover:

```text
0.05, 0.10, 0.15, ..., 0.90
```

Thresholds must also support explicit list and range specifications.

#### FR-6: Generic Primary Metric Selection

Threshold selection must be generic and config-driven.

Selection config should support:

- metric level: `slice` or `volume`
- metric name, using the report metric key
- statistic: `mean` or `median`
- direction: `max` or `min`

For ISLES26 3D threshold analysis, the default primary selection should be volume-level mean Dice using the 3D metric key already reported by the evaluation registry, such as `DiceNativeCoefficient`.

#### FR-7: 3D Metrics

For 3D volume evaluation, compute at minimum:

- Dice, via existing 3D Dice metric implementation.
- Surface Dice, via existing MONAI-backed surface Dice metric implementation.
- HD95, via existing 3D HD95 implementation.
- Predicted volume.
- Ground-truth volume.
- Predicted/ground-truth volume ratio.

Metric class names in reports are acceptable. No shorthand remapping is required for this PRD.

#### FR-8: Per-Case Threshold Records

The pipeline must emit or internally retain per-case/per-threshold metric rows sufficient to compute:

- mean metrics per threshold
- median metrics per threshold
- best global threshold
- per-case oracle threshold
- predicted/GT volume ratio summaries

At minimum, per-case rows should include:

- evaluation level: `slice` or `volume`
- case or sample identifier
- threshold
- configured metric values
- predicted volume where available
- ground-truth volume where available
- predicted/GT volume ratio where available

#### FR-9: Reports and Artifacts

Outputs should include:

- canonical JSON report
- per-threshold aggregate CSV
- per-case/per-threshold CSV
- summary text report
- optional reconstructed volume export where already supported and appropriate

The summary must clearly compare:

- fixed/default threshold, especially `0.5`
- best global threshold
- per-case oracle threshold

#### FR-10: Compatibility With Existing Evaluation Package

- Preserve existing `scripts/evaluation` behavior unless intentionally superseded by the new model evaluation entrypoint.
- Keep current nnU-Net evaluation under `scripts/nnunet` unchanged except for optional reuse of shared helpers.
- Keep existing metric implementations in `src/metrics/metrics.py` as the source of truth.

### 2.2 Non-Functional Requirements

#### NFR-1: Reproducibility

- The final composed evaluation config must be written to the output directory.
- The report must include run directory, checkpoint path, threshold protocol, data subset, validation config, and overrides.
- The pipeline should make it clear whether the evaluation used `val_fast` or `val_full`.

#### NFR-2: Maintainability

- The model evaluation entrypoint should be thin.
- Core logic should live in reusable modules with focused responsibilities.
- The pipeline should follow the typed request/orchestration style already present in `scripts/nnunet/core/evaluation_pipeline.py`.

#### NFR-3: Performance and Memory

- The 3D discriminative path must use sliding-window inference when configured.
- The pipeline should avoid storing full prediction volumes for all cases unless explicitly needed.
- Per-case metric rows are acceptable and expected. Full probability volumes should not be retained across the full validation set by default.

#### NFR-4: Failure Modes

Errors should be explicit for:

- missing run config
- missing checkpoint
- unsupported loader mode
- unsupported 3D diffusion evaluation with the current 2D-shaped diffusion adapters
- unsupported threshold metric selection
- missing primary metric in selected level
- invalid threshold protocol

---

## 3. User Stories and Acceptance Criteria

### Story 1: Evaluate a Model As Trained

As an ML researcher, I want to evaluate a trained model using its saved Hydra config so that I can reproduce the validation behavior associated with the run.

Acceptance criteria:

- Running model evaluation with only run directory and model name succeeds for supported 2D and 3D discriminative configurations.
- The output records the loaded run config and checkpoint path.
- The selected validation subset matches the run config unless overridden.

### Story 2: Override Validation Scope

As an ML researcher, I want to evaluate the same checkpoint on `val_fast` or `val_full` without editing the saved run config.

Acceptance criteria:

- `dataset.active_subsets.val=val_fast` and `dataset.active_subsets.val=val_full` overrides are supported.
- The output metadata states which subset was used.
- Data is loaded through `get_dataloaders(cfg)`.

### Story 3: Fixed Threshold Evaluation

As an ML researcher, I want to evaluate a model at threshold `0.5` to compare with standard validation behavior.

Acceptance criteria:

- `fixed` threshold mode evaluates exactly one threshold.
- Report includes aggregate volume or slice metrics at that threshold.
- For 3D volumes, report includes predicted and ground-truth volume statistics.

### Story 4: Global Threshold Sweep

As an ML researcher, I want to sweep thresholds and identify the best global threshold for a configured primary metric.

Acceptance criteria:

- Thresholds `0.05` through `0.90` are evaluated by default in sweep mode.
- The best global threshold is selected from configured level, metric, statistic, and direction.
- The report compares best global threshold to `0.5` when `0.5` is part of the evaluated set.

### Story 5: Per-Case Oracle Threshold

As an ML researcher, I want a per-case oracle threshold ceiling so I can estimate whether case-specific calibration may matter.

Acceptance criteria:

- For each case, the best threshold is selected by configured primary metric.
- Oracle per-case results are aggregated and reported.
- A per-case table includes the selected oracle threshold for each case.

---

## 4. Components of Interest

### 4.1 Existing Components To Reuse

#### `src.data.loaders`

Use `get_dataloaders(cfg)` and `validate_dataset_contract(cfg)`.

Important behavior:

- `random_patches_3d` trains on random patches but validates on full volumes.
- `dataset.active_subsets.val` selects the validation subset.
- `data_profile` composes the dataset, data mode, and data IO contract.

#### `src.utils.valid_utils`

Use `build_validation_inferer(diffusion, cfg)` for validation inference.

Important behavior:

- Supports direct inference and sliding-window inference.
- Resolves ROI from validation config or dataset preprocessing config.
- Allows 3D discriminative validation to use the same path as training validation.

#### `src.diffusion.discriminative_adapter.DiscriminativeAdapter`

Use `sample(conditioned_image)` as the public probability-producing inference API for discriminative models.

#### `scripts/evaluation/contracts.py`

Reuse `SliceSample`, `VolumeSample`, `ThresholdProtocol`, and running-stat contracts where applicable.

#### `scripts/evaluation/metrics_registry.py`

Reuse threshold-aware 2D metrics for slice evaluation.

#### `scripts/evaluation/metrics_registry_3d.py`

Reuse threshold-aware 3D metrics for volume evaluation.

#### `scripts/evaluation/reporting.py`

Reuse and extend report writers.

#### `scripts/evaluation/threshold_protocol.py`

Reuse threshold parsing and fixed/sweep helpers, but extend protocol semantics.

#### `scripts/nnunet/core/evaluation_pipeline.py`

Use as design precedent for typed request resolution and orchestration.

### 4.2 Existing Components To Deprecate Later

#### `scripts/analysis/threshold_analysis.py`

This should remain for now but should be considered legacy once the new evaluation threshold protocol reaches feature parity.

Deprecation should happen after:

- fixed threshold reports are available in `scripts/evaluation`
- sweep reports are available in `scripts/evaluation`
- visual/plot needs are either ported or intentionally replaced
- current users have migration commands

---

## 5. Necessary Changes

### 5.1 New Configs

Add an evaluation config group for the `scripts/evaluation` package:

```text
configs/evaluation/default.yaml
configs/evaluation/fixed_threshold.yaml
configs/evaluation/threshold_sweep.yaml
configs/evaluation/threshold_sweep_with_oracle.yaml
```

Proposed config shape:

```yaml
evaluation:
  input_source: live_model
  run_dir: null
  model_name: null
  checkpoint:
    use_ema: false
  output_dir: null
  device: null

  levels:
    - volume

  threshold_protocol:
    mode: sweep_with_oracle
    thresholds: "0.05:0.90:0.05"
    fixed_threshold: 0.5
    primary:
      level: volume
      metric: DiceNativeCoefficient
      statistic: mean
      direction: max

  reporting:
    write_per_case_threshold_csv: true
    write_summary_text: true
    write_config: true
```

`evaluation.input_source` identifies where predictions come from. This PRD implements `live_model`, meaning predictions are produced by loading a repository-trained checkpoint and running live inference. Future values could describe file-based or cached prediction sources if the nnU-Net and evaluation frontends are unified later.

The config package should be `configs/evaluation/*`. The implementation should avoid "custom model" naming: repository-trained models are the normal model evaluation target.

### 5.2 Model Evaluation Entrypoint

Add a new entrypoint:

```text
scripts/evaluation/evaluate_model.py
```

Responsibilities:

- Compose or load the evaluation config.
- Load the run config from `evaluation.run_dir`.
- Apply evaluation overrides.
- Call the model evaluation pipeline.
- Print a concise summary and output paths.

Example target command:

```bash
python3 -m scripts.evaluation.evaluate_model \
  evaluation.run_dir=/path/to/run \
  evaluation.model_name=best_model_step_010000_dice_3d_0.6234
```

Override example:

```bash
python3 -m scripts.evaluation.evaluate_model \
  evaluation.run_dir=/path/to/run \
  evaluation.model_name=best_model_step_010000_dice_3d_0.6234 \
  dataset.active_subsets.val=val_full \
  validation=sliding_window_3d_metrics_full \
  evaluation.threshold_protocol.mode=sweep_with_oracle \
  evaluation.threshold_protocol.primary.level=volume \
  evaluation.threshold_protocol.primary.metric=DiceNativeCoefficient
```

### 5.3 Config Loading and Override Utilities

Add:

```text
scripts/evaluation/model_config.py
```

Responsibilities:

- Load `<run_dir>/.hydra/config.yaml`.
- Apply evaluation policy config.
- Apply Hydra-style overrides.
- Write the final resolved config to output artifacts.

The current 2D diffusion/discriminative entrypoint already has partial override logic. That logic should either be moved here or replaced with a cleaner shared implementation.

### 5.4 Model Loading Utilities

Add:

```text
scripts/evaluation/model_loader.py
```

Responsibilities:

- Find checkpoints.
- Build model using existing model factory.
- Build diffusion/discriminative adapter using existing repository APIs.
- Load checkpoint state dict.
- Handle DDP and adapter key prefixes.
- Return an eval-mode model or adapter exposing `sample()`.

This module should extract logic currently embedded in:

```text
scripts/analysis/threshold_analysis.py
scripts/evaluation/compute_segmentation_metrics_for_diffusionmodel_2d_predictions.py
```

### 5.5 Live Model IO Producers

Add or refactor:

```text
scripts/evaluation/io_model_volumes.py
```

Responsibilities:

- Produce `VolumeSample` for 3D validation loaders.
- Normalize model outputs to probability tensors.
- Preserve sample identity and metadata.
- Reject models that cannot satisfy the 3D volume probability contract.

Existing `scripts/evaluation/io_diffusion.py` should be treated as the legacy 2D live-model slice producer despite its diffusion-era name. It already supports both 2D generative diffusion and 2D discriminative models through the shared `model.sample(...)`/`model(image)` inference convention. The primary new IO gap is 3D volume inference, so the first new IO module should be representation-based rather than discriminative-specific.

Proposed APIs:

```python
def iter_model_volume_samples(
    model,
    dataloader,
    cfg,
    device,
    ...
) -> Iterator[VolumeSample]:
    ...
```

For current 3D discriminative models:

- Use `build_validation_inferer(model_or_adapter, cfg)`.
- Iterate over validation batches from `get_dataloaders(cfg)["val"]`.
- For each item, emit a `VolumeSample` with probability prediction volume and GT volume.

For future 3D diffusion models, the same IO module should work once the diffusion adapter contract is upgraded so that `sample(conditioned_image)` returns `[B, C, H, W, D]` probability volumes for `[B, C, H, W, D]` conditioning images.

### 5.6 Evaluation Pipeline

Add:

```text
scripts/evaluation/evaluation_pipeline.py
```

Proposed typed request:

```python
@dataclass(frozen=True)
class ModelEvaluationRequest:
    run_dir: Path
    model_name: str
    output_dir: Path
    device: str
    levels: Sequence[str]
    threshold_protocol: EvaluationThresholdProtocol
    use_ema: bool
    loader_mode: str
    data_dim: str
    diffusion_type: str
```

Responsibilities:

- Resolve request from final config.
- Validate supported mode matrix.
- Build model.
- Build dataloader.
- Dispatch to slice or volume evaluation.
- Build reports.
- Return output paths and summary.

### 5.7 Extended Threshold Protocol

Extend:

```text
scripts/evaluation/threshold_protocol.py
```

Current support:

- fixed threshold
- sweep thresholds
- primary threshold selection from slice metrics

Required extensions:

- structured primary metric selection
- selection from slice or volume level
- mean and median statistic support
- min/max direction support
- oracle per-case selection

Possible new dataclasses:

```python
@dataclass(frozen=True)
class PrimaryMetricSelector:
    level: Literal["slice", "volume"]
    metric: str
    statistic: Literal["mean", "median"]
    direction: Literal["max", "min"]

@dataclass(frozen=True)
class EvaluationThresholdProtocol:
    mode: Literal["fixed", "sweep", "oracle_per_case", "sweep_with_oracle"]
    thresholds: list[float]
    fixed_threshold: float
    primary: PrimaryMetricSelector
```

### 5.8 Per-Case Threshold Records

Add a component such as:

```text
scripts/evaluation/threshold_records.py
```

Responsibilities:

- Store compact per-case/per-threshold rows.
- Compute aggregate statistics.
- Compute global best threshold.
- Compute per-case oracle threshold.
- Write per-case CSVs.

This can be initially implemented for volume-level rows, then extended to slice-level rows as needed.

Proposed output row fields:

```text
level
case_id
threshold
DiceNativeCoefficient
SurfaceDiceMonai
HausdorffDistance95Native
PredictedVolumeMm3
GroundTruthVolumeMm3
pred_gt_volume_ratio
```

The exact metric columns should be dynamic based on configured metric names.

### 5.9 Reporting Extensions

Extend `scripts/evaluation/reporting.py` to include:

- per-case/per-threshold CSV writer
- global threshold comparison block
- oracle summary block
- median statistics in threshold rows
- final resolved config path

Report JSON should include:

```json
{
  "threshold_analysis": {
    "fixed_threshold": {...},
    "best_global_threshold": {...},
    "oracle_per_case": {...},
    "primary_selector": {...}
  }
}
```

### 5.10 Existing Entrypoint Migration

The current entrypoint:

```text
scripts/evaluation/compute_segmentation_metrics_for_diffusionmodel_2d_predictions.py
```

should either:

- remain as a legacy compatibility wrapper, or
- call into the new model evaluation pipeline for supported 2D paths.

Do not remove it in the first implementation.

### 5.11 SLURM Runner

Add:

```text
scripts/evaluation/slurm_runners/run_evaluate_model.py
```

Responsibilities:

- Submit `python3 -m scripts.evaluation.evaluate_model ...` jobs.
- Use flat argparse/config generation on the submission side.
- Reuse `scripts/slurm/utils/commandline_utils.py` where possible.
- Avoid importing Hydra/OmegaConf in the runner, because the cluster submission environment may not have those packages installed.
- Forward model evaluation overrides as command-line `key=value` strings to the Python module executed inside the job container.

---

## 6. Expected Output State

### 6.1 Package Structure

Expected new or modified files:

```text
configs/evaluation/default.yaml
configs/evaluation/fixed_threshold.yaml
configs/evaluation/threshold_sweep.yaml
configs/evaluation/threshold_sweep_with_oracle.yaml

scripts/evaluation/evaluate_model.py
scripts/evaluation/slurm_runners/run_evaluate_model.py
scripts/evaluation/model_config.py
scripts/evaluation/model_loader.py
scripts/evaluation/io_model_volumes.py
scripts/evaluation/evaluation_pipeline.py
scripts/evaluation/threshold_records.py

scripts/evaluation/threshold_protocol.py      # extend
scripts/evaluation/reporting.py               # extend
scripts/evaluation/metrics_engine.py           # extend or complement
```

### 6.2 Example Output Directory

```text
<run_dir>/analysis/evaluation_v3/<model_name>_<timestamp>/
├── canonical_results.json
├── evaluation_summary.txt
├── resolved_evaluation_config.yaml
├── volume_metrics_per_threshold.csv
├── per_case_threshold_metrics.csv
├── oracle_per_case_thresholds.csv
└── reconstructed_volumes/                 # optional
```

### 6.3 Summary Report Content

The text summary should answer:

- Which run and checkpoint were evaluated?
- Which validation subset was used?
- Which data mode and validation inference mode were used?
- Which thresholds were evaluated?
- What was the metric at threshold `0.5`?
- What was the best global threshold and metric value?
- What was the per-case oracle aggregate?
- How large is the gap:
  - best global minus `0.5`
  - oracle minus best global

### 6.4 Scientific Interpretation

The output should support the following interpretation:

- If best global threshold strongly beats `0.5`, global threshold calibration is likely useful.
- If oracle strongly beats best global threshold, case-specific calibration or volume-bias correction may matter.
- If neither improves materially, thresholding is probably not the limiting factor.

---

## 7. Risks and Mitigations

### 7.1 Risk: Metric Mismatch With Training Validation

Risk:

- Evaluation at threshold `0.5` may not exactly match training validation metrics if the metric registry or inference path differs.

Mitigation:

- Reuse `build_validation_inferer()`.
- Reuse `get_dataloaders(cfg)`.
- Use `src/metrics/metrics.py` implementations.

### 7.2 Risk: 3D Metric Runtime Is Expensive

Risk:

- Surface Dice and HD95 can be slow on full volumes and threshold sweeps multiply the cost.

Mitigation:

- Keep `validation=sliding_window_3d_metrics_subset` as a fast default.
- Allow metric list configuration.
- Allow users to enable full metric bundles only for final analysis.

### 7.3 Risk: Oracle Requires More State Than Current Streaming Engine

Risk:

- Current streaming aggregates are not enough to compute oracle and median.

Mitigation:

- Store compact scalar per-case/per-threshold records rather than full prediction volumes.
- Keep full probability volumes out of long-term memory.

### 7.4 Risk: Config Composition Complexity

Risk:

- Loading a saved run config and then applying new group overrides may be more complex than normal Hydra startup.

Mitigation:

- Implement a focused config utility with tests.
- Persist final resolved config.
- Keep CLI examples explicit.

### 7.5 Risk: 3D Diffusion Capability Is Confusing

Risk:

- Users may expect 3D diffusion to work because 2D diffusion and 3D discriminative evaluation work.
- The new 3D IO module can be representation-based in design, but the current non-discriminative diffusion adapters are not 3D-ready. Current sampling code constructs 2D masks with shape `[B, C, H, W]` from scalar `image_size`, while 3D volume evaluation requires probability tensors shaped like `[B, C, H, W, D]`.

Mitigation:

- Fail early with a clear message:
  - detected `data_mode.dim=3d`
  - detected non-discriminative `diffusion.type`
  - explain that current diffusion adapters do not satisfy the 3D volume inference contract
- Keep `io_model_volumes.py` generic by contract, so future 3D diffusion can plug in without renaming the IO layer.
- Document the future 3D diffusion contract in `src/diffusion/diffusion.py`.

Future 3D diffusion support should require:

- A 3D-capable architecture whose forward pass accepts noisy masks, timesteps, and conditioning volumes with 5D tensors.
- Diffusion adapters whose `sample(conditioned_image)` returns probability volumes with the same batch and spatial rank as the target, e.g. `[B, C, H, W, D]`.
- Training and validation configs that define 3D mask shape/ROI semantics explicitly instead of relying on scalar 2D `image_size`.
- A clearly chosen inference policy for full-volume versus patch/sliding-window diffusion sampling.

---

## 8. Assumptions and Dependencies

### 8.1 Assumptions

- Run directories contain `.hydra/config.yaml`.
- Checkpoints are compatible with the current model factory and adapter code.
- ISLES26 3D validation is available through existing dataloaders.
- 3D discriminative models expose probability inference through `DiscriminativeAdapter.sample()`.
- Current non-discriminative diffusion adapters are 2D-shaped and do not yet satisfy the 3D volume inference contract.
- Current binary segmentation tasks use sigmoid probabilities and thresholded masks.

### 8.2 Dependencies

Internal dependencies:

- `src.data.loaders`
- `src.utils.valid_utils`
- `src.models.model_factory`
- `src.diffusion.diffusion`
- `src.diffusion.discriminative_adapter`
- `src.metrics.metrics`
- `scripts.evaluation.metrics_registry`
- `scripts.evaluation.metrics_registry_3d`
- `scripts.evaluation.reporting`
- `scripts.evaluation.threshold_protocol`

External dependencies:

- PyTorch
- MONAI
- OmegaConf / Hydra
- NumPy
- nibabel, only for existing volume export or file-based paths

---

## 9. Testing Plan

### 9.1 Unit Tests

Add tests for:

- threshold protocol parsing
- primary selector resolution
- threshold selection from slice results
- threshold selection from volume results
- median statistic selection
- min/max direction handling
- oracle per-case threshold selection
- predicted/GT volume ratio handling
- unsupported 3D diffusion capability validation error

### 9.2 Integration Tests

Add mocked or lightweight integration tests for:

- config loading from a synthetic run directory
- applying overrides to the run config
- 2D repository-model slice evaluation using mocked inference samples
- 3D discriminative volume evaluation using mocked dataloader/model outputs
- report artifact creation

### 9.3 Manual Validation

Run on a known ISLES26 3D DynUNet checkpoint:

1. Fixed threshold `0.5` on `val_fast`.
2. Sweep with oracle on `val_fast`.
3. Sweep with oracle on `val_full`.
4. Compare threshold `0.5` Dice against training validation logs where available.

---

## 10. Resolved Planning Decisions

1. Default output directory: `<run_dir>/analysis/evaluation_v3/`.
2. Median support in the first implementation is required for primary selection and top-level summaries, not every report field.
3. The first implementation focuses on JSON/CSV/text outputs. Plotting can be ported after the model evaluation path is validated.
4. `compute_segmentation_metrics_for_diffusionmodel_2d_predictions.py` remains untouched until the new entrypoint is validated.
5. Live-model IO is split by representation. Existing `io_diffusion.py` remains the legacy 2D slice producer; new `io_model_volumes.py` handles 3D volume samples.
6. Current 3D non-discriminative diffusion is rejected with a capability error because current diffusion adapters are 2D-shaped. Future 3D-capable diffusion should plug into `io_model_volumes.py` after satisfying the volume inference contract.
7. SLURM evaluation runners use flat argparse/ersatz config generation via `scripts/slurm/utils/commandline_utils.py` and must not require Hydra/OmegaConf on the cluster submission side.
---

## 11. Approval Criteria

This PRD is ready for implementation planning once the following are agreed:

- Repository-model evaluation remains live-inference based.
- nnU-Net file-based evaluation remains separate.
- Existing repo dataloaders and validation inferers are the only source for volume loading/preprocessing.
- Threshold analysis lives under `scripts/evaluation`.
- 3D diffusion evaluation is explicitly unsupported in this phase because current diffusion adapters are 2D-shaped; the 3D IO design remains generic for future 3D-capable diffusion adapters.
- Per-case/per-threshold records are acceptable as required artifacts for oracle analysis.

