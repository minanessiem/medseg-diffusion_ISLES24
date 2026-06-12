# Change Action Plan: Config-Driven Custom Model Evaluation and Thresholding

**Document Version:** 1.0  
**Date:** 2026-06-11  
**PRD Reference:** `docs/PRD_Config_Driven_Custom_Model_Evaluation_Thresholding.md`  
**Status:** Draft - Pending Review  
**Scope:** Implement a config-driven repository-model live inference evaluation pipeline with fixed, sweep, and per-case oracle threshold analysis.

---

## 1. Overall Plan Summary

This CAP implements the PRD by introducing a new config-driven repository-model evaluation path under `scripts/evaluation`, while preserving the existing nnU-Net file-based evaluation path under `scripts/nnunet`.

The implementation will:

1. Add evaluation configs for repository-model live-inference evaluation.
2. Extract and centralize run-config loading, overrides, checkpoint discovery, and model loading.
3. Add live-model IO producers by representation: legacy 2D slice IO plus new 3D volume IO.
4. Extend threshold protocol handling to support fixed, sweep, oracle, and sweep-with-oracle modes.
5. Add compact per-case/per-threshold records for global threshold selection, median summaries, oracle analysis, and volume-ratio reporting.
6. Add a typed model evaluation pipeline and a thin `python3 -m scripts.evaluation.evaluate_model` entrypoint.
7. Add focused tests and update documentation.

The first implementation should prioritize correctness, reproducibility, and clear output artifacts over plotting or visual presentation. Plotting from legacy `scripts/analysis/threshold_analysis.py` can be ported later after JSON/CSV/text outputs are validated.

---

## 2. Implementation Principles

- Keep nnU-Net file-based evaluation separate and unchanged unless a shared helper can be reused safely.
- Reuse `src.data.loaders.get_dataloaders(cfg)` for all repository-model data access.
- Reuse `src.utils.valid_utils.build_validation_inferer()` for validation inference behavior, especially 3D sliding-window inference.
- Reuse `src.metrics.metrics.py` implementations through existing evaluation registries.
- Store scalar per-case/per-threshold records, not full prediction volumes.
- Preserve existing entrypoints until the new path is validated.
- Fail explicitly for current 3D non-discriminative diffusion volume evaluation because the existing diffusion adapters are 2D-shaped; keep the 3D IO module generic for future 3D-capable diffusion adapters.
- Keep each new module focused on one responsibility.

---

## 3. Phase 0: Baseline Audit and Guardrails

### Goal

Confirm the current behavior and protect existing entrypoints before adding the new path.

### Subtasks

#### 0.1 Review Existing Tests and Entrypoints

Files of interest:

- `scripts/evaluation/compute_segmentation_metrics_for_diffusionmodel_2d_predictions.py`
- `scripts/evaluation/compute_segmentation_metrics_for_nnunet_2d_predictions.py`
- `scripts/evaluation/threshold_protocol.py`
- `scripts/evaluation/metrics_engine.py`
- `scripts/evaluation/reporting.py`
- `scripts/evaluation/metrics_registry_3d.py`
- `scripts/nnunet/core/evaluation_pipeline.py`
- `tests/test_evaluation_*.py`

Actions:

- Identify tests that lock existing behavior.
- Note where old 2D diffusion/discriminative evaluation code should be extracted instead of rewritten.
- Confirm no nnU-Net behavior needs modification in this implementation.

### Validation

Run the current evaluation test subset before making behavioral changes:

```bash
python3 -m pytest tests/test_evaluation_threshold_protocol.py tests/test_evaluation_reporting.py tests/test_evaluation_metrics_registry_3d.py tests/test_evaluation_dual_level_engine.py
```

### Rollback

No code changes in this phase.

---

## 4. Phase 1: Evaluation Config Schema

### Goal

Introduce config files for model evaluation policy without changing runtime behavior.

### Files to Add

```text
configs/evaluation/default.yaml
configs/evaluation/fixed_threshold.yaml
configs/evaluation/threshold_sweep.yaml
configs/evaluation/threshold_sweep_with_oracle.yaml
```

### Proposed Defaults

`configs/evaluation/default.yaml`:

```yaml
# @package _global_

evaluation:
  input_source: live_model
  run_dir: null
  model_name: null
  output_dir: null
  device: null

  checkpoint:
    use_ema: false

  levels:
    - volume

  threshold_protocol:
    mode: fixed
    thresholds: "0.5"
    fixed_threshold: 0.5
    primary:
      level: volume
      metric: DiceNativeCoefficient
      statistic: mean
      direction: max

  reporting:
    write_json: true
    write_summary_text: true
    write_per_threshold_csv: true
    write_per_case_threshold_csv: true
    write_config: true
    export_reconstructed_volumes: false
    max_export_volumes_per_case: null
```

`evaluation.input_source` identifies where predictions come from. The first implementation supports `live_model`, meaning a repository checkpoint is loaded and evaluated through live inference. Future values can represent file-based or cached prediction sources if the nnU-Net and evaluation frontends are unified later.

Preset intent:

- `fixed_threshold.yaml`: one threshold, default `0.5`.
- `threshold_sweep.yaml`: `0.05:0.90:0.05`, global best threshold.
- `threshold_sweep_with_oracle.yaml`: same sweep plus per-case oracle.

### Subtasks

#### 1.1 Add Config Group

Create the config files above.

#### 1.2 Align Naming With Hydra Patterns

The group should move to a simpler `evaluation` group.

Recommended initial path:

```text
configs/evaluation/*.yaml
```

This avoids colliding with current `configs/validation/*` semantics.

#### 1.3 Document Example Overrides

Add examples in either the CAP appendix or later README updates:

```bash
python3 -m scripts.evaluation.evaluate_model \
  evaluation.run_dir=/path/to/run \
  evaluation.model_name=best_model \
  dataset.active_subsets.val=val_full \
  validation=sliding_window_3d_metrics_full \
  evaluation.threshold_protocol.mode=sweep_with_oracle
```

### Validation

Config files should parse with OmegaConf/Hydra once the entrypoint exists. Before that, they can be inspected manually.

### Rollback

Remove the new config files.

---

## 5. Phase 2: Config Loading and Override Utilities

### Goal

Build the mechanism that loads a trained run config, merges evaluation policy, and applies Hydra-style overrides.

### Files to Add

```text
scripts/evaluation/model_config.py
```

### Files to Reference

- `scripts/evaluation/compute_segmentation_metrics_for_diffusionmodel_2d_predictions.py`
- `scripts/nnunet/evaluate_nnunet_results.py`
- `scripts/slurm/single_job_runner.py`
- `start_training.py`

### Subtasks

#### 2.1 Implement Run Config Loading

Add:

```python
def load_run_config(run_dir: Path) -> DictConfig:
    ...
```

Responsibilities:

- Load `<run_dir>/.hydra/config.yaml`.
- Raise a clear `FileNotFoundError` when missing.
- Return a mutable `DictConfig` suitable for evaluation overrides.

#### 2.2 Implement Evaluation Config Merge

Add:

```python
def merge_evaluation_config(run_cfg: DictConfig, eval_cfg: DictConfig) -> DictConfig:
    ...
```

Responsibilities:

- Preserve run config as the base.
- Merge `evaluation.*` config into the base.
- Avoid clobbering training/data config unless explicit overrides request it.

#### 2.3 Implement Hydra-Style Overrides

Add:

```python
def apply_evaluation_overrides(cfg: DictConfig, overrides: Sequence[str]) -> DictConfig:
    ...
```

Support:

- dotted key overrides, e.g. `dataset.active_subsets.val=val_full`
- top-level group overrides when practical, e.g. `validation=sliding_window_3d_metrics_full`
- scalar/list parsing via YAML/OmegaConf parsing

This can adapt the partial override logic currently in the legacy 2D diffusion/discriminative entrypoint.

#### 2.4 Resolve Output Directory

Add:

```python
def resolve_evaluation_output_dir(cfg: DictConfig, timestamp: str) -> Path:
    ...
```

Default output:

```text
<run_dir>/analysis/evaluation_v3/<model_name>_<timestamp>/
```

Resolved decision:

- Use `<run_dir>/analysis/evaluation_v3/` for the new repository-model evaluation path.

#### 2.5 Persist Resolved Config

Add:

```python
def write_resolved_evaluation_config(cfg: DictConfig, output_dir: Path) -> Path:
    ...
```

Output:

```text
resolved_evaluation_config.yaml
```

### Validation

Add tests:

```text
tests/test_evaluation_model_config.py
```

Test cases:

- missing `.hydra/config.yaml` raises clear error
- run config loads from synthetic directory
- `dataset.active_subsets.val=val_full` applies correctly
- `validation=sliding_window_3d_metrics_full` group override resolves correctly
- final config includes `evaluation.*`

Run:

```bash
python3 -m pytest tests/test_evaluation_model_config.py
```

### Rollback

Remove `model_config.py` and its tests. No existing runtime should be affected.

---

## 6. Phase 3: Model and Adapter Loading Utilities

### Goal

Extract repository-model checkpoint loading from legacy scripts into a reusable module.

### Files to Add

```text
scripts/evaluation/model_loader.py
```

### Files to Reference

- `scripts/analysis/threshold_analysis.py`
- `scripts/evaluation/compute_segmentation_metrics_for_diffusionmodel_2d_predictions.py`
- `src.models.model_factory`
- `src.diffusion.diffusion`
- `src.diffusion.discriminative_adapter`

### Subtasks

#### 3.1 Implement Checkpoint Discovery

Add:

```python
def find_checkpoint(run_dir: Path, model_name: str, use_ema: bool = False) -> Path:
    ...
```

Search order:

```text
<run_dir>/models/best/
<run_dir>/models/checkpoints/
<run_dir>/models/
```

Behavior:

- Strip `.pth` suffix if supplied.
- Support existing EMA lookup pattern.
- Raise a clear error listing searched paths.

#### 3.2 Implement Model/Adapter Construction

Add:

```python
def build_model_for_evaluation(cfg: DictConfig, checkpoint_path: Path, device: str) -> nn.Module:
    ...
```

Responsibilities:

- Build model through existing model factory.
- Wrap with diffusion/discriminative adapter through existing APIs.
- Move to device.
- Set eval mode.

#### 3.3 Preserve State Dict Compatibility

Support current checkpoint structures:

- raw state dict
- `model_state_dict`
- `state_dict`

Prefer `src.training.checkpoint_utils.load_model_state_dict_compat()` for checkpoint compatibility. Preserve support for:

- `module.`
- `wrapped_model.base_model.`
- `model.model.`
- `model.`

#### 3.4 Expose Model Kind Helpers

Add:

```python
def resolve_diffusion_type(cfg: DictConfig) -> str:
    ...

def is_discriminative_config(cfg: DictConfig) -> bool:
    ...
```

These helpers will be used by the mode matrix validation.

### Validation

Add tests:

```text
tests/test_evaluation_model_loader.py
```

Test cases:

- checkpoint discovery in each supported directory
- `.pth` suffix normalization
- missing checkpoint error includes searched paths
- state dict prefix stripping with a tiny dummy model
- discriminative type detection

Run:

```bash
python3 -m pytest tests/test_evaluation_model_loader.py
```

### Rollback

Remove `model_loader.py` and tests. Legacy scripts continue using their embedded logic.

---

## 7. Phase 4: Extended Threshold Protocol

### Goal

Generalize threshold protocol handling for slice and volume levels, global threshold selection, median statistics, and oracle modes.

### Files to Modify

```text
scripts/evaluation/threshold_protocol.py
scripts/evaluation/contracts.py
```

### Files to Add If Needed

```text
scripts/evaluation/threshold_selection.py
```

### Subtasks

#### 4.1 Add Structured Primary Selector

Add dataclass:

```python
@dataclass(frozen=True)
class PrimaryMetricSelector:
    level: Literal["slice", "volume"]
    metric: str
    statistic: Literal["mean", "median"]
    direction: Literal["max", "min"]
```

#### 4.2 Add Extended Protocol Dataclass

Add dataclass:

```python
@dataclass(frozen=True)
class EvaluationThresholdProtocol:
    mode: Literal["fixed", "sweep", "oracle_per_case", "sweep_with_oracle"]
    thresholds: list[float]
    fixed_threshold: float
    primary: PrimaryMetricSelector
```

Keep existing `ThresholdProtocol` if needed for backward compatibility.

#### 4.3 Parse Protocol From Config

Add:

```python
def build_evaluation_threshold_protocol(cfg: DictConfig) -> EvaluationThresholdProtocol:
    ...
```

Behavior:

- Parse `evaluation.threshold_protocol.thresholds`.
- Include `fixed_threshold` in threshold set when needed for comparison.
- Validate mode.
- Validate primary selector.

#### 4.4 Preserve Legacy Helpers

Existing functions such as `make_fixed_protocol`, `make_sweep_protocol_from_spec`, and `select_primary_threshold` should remain usable by current tests and legacy entrypoints.

If new code needs different behavior, add new functions instead of changing old semantics destructively.

#### 4.5 Generic Selection Helpers

Add helpers that operate on aggregate threshold rows:

```python
def select_best_threshold_from_rows(rows, selector: PrimaryMetricSelector) -> float:
    ...
```

### Validation

Update and add tests:

```text
tests/test_evaluation_threshold_protocol.py
tests/test_evaluation_threshold_selection.py
```

Test cases:

- parse `fixed`
- parse `sweep`
- parse `oracle_per_case`
- parse `sweep_with_oracle`
- include fixed threshold in sweep comparison when missing
- select max mean
- select max median
- select min mean
- reject unsupported level/statistic/direction

Run:

```bash
python3 -m pytest tests/test_evaluation_threshold_protocol.py tests/test_evaluation_threshold_selection.py
```

### Rollback

Revert modifications to `threshold_protocol.py` and `contracts.py`, remove new selection module/tests. Existing tests should remain the safety check.

---

## 8. Phase 5: Per-Case Threshold Records

### Goal

Add compact scalar records that make global threshold selection, median summaries, oracle thresholding, and volume-ratio reporting possible.

### Files to Add

```text
scripts/evaluation/threshold_records.py
```

### Files to Modify

```text
scripts/evaluation/reporting.py
```

### Subtasks

#### 5.1 Define Record Dataclasses

Add:

```python
@dataclass(frozen=True)
class ThresholdMetricRecord:
    level: str
    case_id: str
    threshold: float
    metrics: Mapping[str, float]
    metadata: Mapping[str, object]
```

Optional specialized aliases can be added later, but start with one flexible record type.

#### 5.2 Compute Volume Ratio

Add helper:

```python
def add_volume_ratio(metrics: Mapping[str, float]) -> dict[str, float]:
    ...
```

Behavior:

- Use `PredictedVolumeMm3` and `GroundTruthVolumeMm3` if available.
- Add `pred_gt_volume_ratio`.
- Handle zero GT volume explicitly.

Recommended zero-GT policy:

- If GT volume is zero and predicted volume is zero, ratio is `1.0`.
- If GT volume is zero and predicted volume is nonzero, ratio is `inf`.
- Preserve this in JSON/CSV with clear representation.

#### 5.3 Aggregate Records Per Threshold

Add:

```python
def aggregate_threshold_records(records, selector_level: str) -> dict[float, dict[str, object]]:
    ...
```

For each threshold and metric:

- count
- mean
- median
- std
- min
- max

#### 5.4 Compute Best Global Threshold

Add:

```python
def select_global_threshold(records, selector: PrimaryMetricSelector) -> dict[str, object]:
    ...
```

Return:

- selected threshold
- selector details
- selected statistic value
- full metrics at that threshold

#### 5.5 Compute Oracle Per Case

Add:

```python
def select_oracle_thresholds(records, selector: PrimaryMetricSelector) -> tuple[list[dict], dict]:
    ...
```

Return:

- per-case selected threshold rows
- aggregate oracle summary

#### 5.6 CSV Writers

Extend `reporting.py` or keep writers in `threshold_records.py`:

```python
def write_per_case_threshold_csv(records, output_dir: Path, filename: str = "per_case_threshold_metrics.csv") -> Path:
    ...

def write_oracle_threshold_csv(rows, output_dir: Path, filename: str = "oracle_per_case_thresholds.csv") -> Path:
    ...
```

### Validation

Add tests:

```text
tests/test_evaluation_threshold_records.py
```

Test cases:

- aggregate mean/median/std by threshold
- global max mean selection
- global max median selection
- oracle per-case selection
- volume ratio normal case
- zero-GT volume ratio policy
- CSV writer creates dynamic metric columns

Run:

```bash
python3 -m pytest tests/test_evaluation_threshold_records.py
```

### Rollback

Remove `threshold_records.py`, revert reporting additions, remove tests.

---

## 9. Phase 6: Live Model IO By Representation

### Goal

Add live inference producers for repository model outputs that yield evaluation contracts while reusing existing dataloaders and validation inferers. The IO boundary should follow representation dimensionality rather than training technique.

### Files to Add

```text
scripts/evaluation/io_model_volumes.py
```

### Files to Reference

- `scripts/evaluation/io_diffusion.py`
- `scripts/evaluation/contracts.py`
- `src.data.loaders`
- `src.utils.valid_utils`
- `src.diffusion.discriminative_adapter`
- `src.diffusion.diffusion`

### Subtasks

#### 6.1 Add Mode Matrix and Capability Validation

Add:

```python
def validate_model_evaluation_mode(cfg: DictConfig) -> None:
    ...
```

Supported:

- `data_mode.dim=2d`, diffusion or discriminative, through existing 2D slice IO
- `data_mode.dim=3d`, `diffusion.type=Discriminative`

Unsupported:

- `data_mode.dim=3d`, non-discriminative diffusion with current adapters

Rationale:

- Existing `io_diffusion.py` is diffusion-era naming, but it is effectively a 2D live-model slice producer because it calls `model.sample(...)` when available and otherwise falls back to `model(image)`.
- Current non-discriminative diffusion adapters construct 2D sample tensors like `[B, C, H, W]` from scalar `image_size`.
- 3D volume IO requires model outputs shaped like the target volume, e.g. `[B, C, H, W, D]`.
- Future 3D diffusion should plug into `io_model_volumes.py` once the diffusion adapter contract is upgraded.

#### 6.2 Preserve Existing 2D Slice Producer

Use existing:

```python
iter_diffusion_case_slice_samples(...)
```

Responsibilities:

- Keep current 2D live-model behavior intact.
- Support 2D diffusion and 2D discriminative models.
- Treat `io_diffusion.py` as legacy 2D live-model slice IO until a later rename or compatibility wrapper is intentionally introduced.

#### 6.3 Implement 3D Volume Producer

Add:

```python
def iter_model_volume_samples(model, dataloader, cfg, device, max_samples=None) -> Iterator[VolumeSample]:
    ...
```

Responsibilities:

- Use `build_validation_inferer(diffusion=model, cfg=cfg)`.
- Iterate `dataloaders["val"]`.
- Run direct or sliding-window inference according to config.
- Emit `VolumeSample` with prediction probabilities and GT volume.
- Include metadata:
  - case ID
  - source sample ID
  - data mode
  - validation inference mode
  - subset if resolvable
  - shape

#### 6.4 Probability Normalization

Add shared helper:

```python
def normalize_probability_prediction(prediction: Tensor) -> Tensor:
    ...
```

Behavior:

- If values are outside `[0, 1]`, apply sigmoid.
- Clamp final output to `[0, 1]`.
- Keep tensors on CPU for metric computation.

#### 6.5 Batch Identity Handling

Add helper for batch sample IDs:

```python
def resolve_batch_item_identity(sample_ids, batch_index: int, item_index: int) -> str:
    ...
```

For 3D volumes, `ISLES26Dataset3D` returns `caseID`, so that should become `case_id` and `volume_id`.

### Validation

Add tests:

```text
tests/test_evaluation_io_model_volumes.py
```

Test cases:

- unsupported current 3D diffusion raises clear capability error
- 3D discriminative dummy model emits `VolumeSample`
- prediction logits get sigmoid-normalized
- probability predictions are clamped
- sample IDs are preserved

Run:

```bash
python3 -m pytest tests/test_evaluation_io_model_volumes.py
```

### Rollback

Remove `io_model_volumes.py` and tests.

---

## 10. Phase 7: Evaluation Pipeline

### Goal

Implement the orchestration layer that resolves the request, runs live inference, evaluates metrics/thresholds, and writes artifacts.

### Files to Add

```text
scripts/evaluation/evaluation_pipeline.py
```

### Files to Modify

```text
scripts/evaluation/reporting.py
scripts/evaluation/metrics_engine.py       # only if needed
```

### Subtasks

#### 7.1 Define Typed Request

Add:

```python
@dataclass(frozen=True)
class ModelEvaluationRequest:
    run_dir: Path
    model_name: str
    checkpoint_path: Path
    output_dir: Path
    device: str
    levels: Sequence[str]
    threshold_protocol: EvaluationThresholdProtocol
    use_ema: bool
    loader_mode: str
    data_dim: str
    diffusion_type: str
```

#### 7.2 Resolve Request From Config

Add:

```python
def build_model_evaluation_request(cfg: DictConfig) -> ModelEvaluationRequest:
    ...
```

Validate:

- `evaluation.run_dir`
- `evaluation.model_name`
- `evaluation.threshold_protocol`
- `evaluation.levels`
- supported mode matrix

#### 7.3 Implement Slice Evaluation

Add:

```python
def evaluate_model_slices(...):
    ...
```

Use existing:

- `DualLevelStreamingMetricsEngine` for slice-level and reconstructed-volume metrics where applicable.
- `threshold_records` for per-case/per-threshold rows if slice-level oracle is enabled.

Initial priority:

- Preserve behavior of current 2D live-model entrypoint.
- Add generic selection/reporting over existing results.

#### 7.4 Implement Volume Evaluation

Add:

```python
def evaluate_model_volumes(...):
    ...
```

Algorithm:

1. Iterate `VolumeSample` predictions from `io_model_volumes.iter_model_volume_samples`.
2. For each volume and threshold:
   - compute configured 3D metrics via `compute_metrics_3d_at_threshold`
   - add volume ratio
   - append `ThresholdMetricRecord`
3. Aggregate records per threshold.
4. Select best global threshold if protocol requires it.
5. Select oracle thresholds if protocol requires it.
6. Build report payload.

#### 7.5 Build Canonical Payload

Extend `build_report_payload()` or add a model-evaluation payload builder:

```python
def build_model_evaluation_payload(...):
    ...
```

Payload should include:

- metadata
- data summary
- protocol
- metrics
- threshold analysis block
- artifact paths where appropriate

#### 7.6 Write Artifacts

Write:

```text
canonical_results.json
evaluation_summary.txt
resolved_evaluation_config.yaml
volume_metrics_per_threshold.csv
per_case_threshold_metrics.csv
oracle_per_case_thresholds.csv
```

Only write oracle CSV when oracle is enabled.

#### 7.7 Return Result Object

Return dict with:

- output directory
- paths
- summary text
- selected global threshold
- oracle summary if present

### Validation

Add tests:

```text
tests/test_evaluation_pipeline.py
```

Test cases:

- request resolution validates required fields
- unsupported current 3D diffusion is rejected with a capability error
- mocked 3D volume samples produce expected CSV/JSON outputs
- fixed protocol writes one threshold
- sweep protocol writes multiple thresholds and global best
- sweep-with-oracle writes oracle output

Run:

```bash
python3 -m pytest tests/test_evaluation_pipeline.py
```

### Rollback

Remove `evaluation_pipeline.py`, revert reporting/engine changes, remove tests.

---

## 11. Phase 8: Config-Driven Entrypoint

### Goal

Add the user-facing model evaluation command.

### Files to Add

```text
scripts/evaluation/evaluate_model.py
scripts/evaluation/slurm_runners/run_evaluate_model.py
```

### Subtasks

#### 8.1 Implement Hydra Entrypoint

Preferred style:

```python
@hydra.main(config_path="../../configs", config_name="evaluation/default", version_base=None)
def main(cfg: DictConfig) -> None:
    ...
```

Responsibilities:

- Load and merge run config.
- Apply CLI overrides if a hybrid Hydra/manual parsing approach is needed.
- Run `run_model_evaluation(cfg)`.
- Print outputs.

Open implementation detail:

- Pure Hydra entrypoint may make "load run config first, then apply command-line overrides to that config" non-trivial.
- If needed, use an argparse wrapper that accepts config names/overrides similarly to `scripts/nnunet/evaluate_nnunet_results.py`.

Preferred user command should still use module form:

```bash
python3 -m scripts.evaluation.evaluate_model ...
```

#### 8.2 Support Minimal Invocation

Required:

```bash
python3 -m scripts.evaluation.evaluate_model \
  evaluation.run_dir=/path/to/run \
  evaluation.model_name=best_model
```

#### 8.3 Support Overrides

Examples:

```bash
python3 -m scripts.evaluation.evaluate_model \
  evaluation.run_dir=/path/to/run \
  evaluation.model_name=best_model \
  dataset.active_subsets.val=val_full \
  validation=sliding_window_3d_metrics_full \
  evaluation.threshold_protocol.mode=sweep_with_oracle
```

#### 8.4 Exit Codes and Errors

Behavior:

- Return nonzero on configuration, checkpoint, mode, or runtime errors.
- Print concise error messages.
- Do not swallow tracebacks in tests.

#### 8.5 Implement SLURM Runner

Add `scripts/evaluation/slurm_runners/run_evaluate_model.py`.

Responsibilities:

- Build a `python3 -m scripts.evaluation.evaluate_model ...` command string.
- Use flat argparse/config generation on the submission side.
- Reuse `scripts/slurm/utils/commandline_utils.py` where possible.
- Do not import Hydra/OmegaConf in the runner, because the cluster submission environment may not have those packages installed.
- Forward evaluation overrides as command-line `key=value` strings to the module executed inside the job container.

### Validation

Add tests:

```text
tests/test_evaluation_model_entrypoint.py
```

Test cases:

- parser/Hydra invocation accepts minimal config
- dry mocked pipeline call gets expected config
- missing run_dir fails
- overrides are forwarded
- SLURM runner dry-run emits the expected `python3 -m scripts.evaluation.evaluate_model ...` command without importing OmegaConf

Run:

```bash
python3 -m pytest tests/test_evaluation_model_entrypoint.py
```

Manual help check:

```bash
python3 -m scripts.evaluation.evaluate_model --help
python3 -m scripts.evaluation.slurm_runners.run_evaluate_model --help
```

### Rollback

Remove `evaluate_model.py`, `run_evaluate_model.py`, and tests. Internal modules remain reusable.

---

## 12. Phase 9: Legacy Compatibility and Deprecation Markers

### Goal

Keep current evaluation scripts working while making the new model evaluation path discoverable.

### Files to Modify

```text
scripts/evaluation/README.md
scripts/analysis/README.md
```

Optional later:

```text
scripts/evaluation/compute_segmentation_metrics_for_diffusionmodel_2d_predictions.py
```

### Subtasks

#### 9.1 Update Evaluation README

Document:

- new model evaluation entrypoint
- fixed threshold mode
- sweep mode
- sweep-with-oracle mode
- `val_fast` / `val_full` override examples
- unsupported 3D diffusion caveat

#### 9.2 Mark Analysis Threshold Script As Legacy

Update `scripts/analysis/README.md` to say:

- `scripts/analysis/threshold_analysis.py` is legacy.
- New threshold analysis should use `scripts.evaluation.evaluate_model`.
- Removal should happen only after feature parity and migration.

#### 9.3 Optional Compatibility Wrapper

After the new path is validated, consider updating:

```text
scripts/evaluation/compute_segmentation_metrics_for_diffusionmodel_2d_predictions.py
```

Options:

- Leave untouched.
- Add deprecation note in docstring.
- Internally call the new pipeline for compatible 2D cases.

Recommendation for first implementation:

- Leave the old entrypoint untouched except documentation.

### Validation

Documentation-only validation plus command help checks:

```bash
python3 -m scripts.evaluation.evaluate_model --help
python3 -m scripts.evaluation.compute_segmentation_metrics_for_diffusionmodel_2d_predictions --help
```

### Rollback

Revert README/doc changes.

---

## 13. Phase 10: End-to-End Validation on ISLES26 3D DynUNet

### Goal

Validate the scientific use case that motivated the work.

### Prerequisites

- Completed ISLES26 3D DynUNet run directory.
- Checkpoint name.
- Accessible data paths from run config or evaluation overrides.

### Subtasks

#### 10.1 Fixed Threshold on `val_fast`

Run:

```bash
python3 -m scripts.evaluation.evaluate_model \
  evaluation.run_dir=<RUN_DIR> \
  evaluation.model_name=<MODEL_NAME> \
  dataset.active_subsets.val=val_fast \
  evaluation.threshold_protocol.mode=fixed \
  evaluation.threshold_protocol.fixed_threshold=0.5
```

Verify:

- output artifacts written
- volume metrics present
- threshold `0.5` Dice is plausible relative to training validation

#### 10.2 Sweep With Oracle on `val_fast`

Run:

```bash
python3 -m scripts.evaluation.evaluate_model \
  evaluation.run_dir=<RUN_DIR> \
  evaluation.model_name=<MODEL_NAME> \
  dataset.active_subsets.val=val_fast \
  evaluation.threshold_protocol.mode=sweep_with_oracle \
  evaluation.threshold_protocol.thresholds=0.05:0.90:0.05
```

Verify:

- best global threshold reported
- oracle summary reported
- per-case threshold CSV present

#### 10.3 Sweep With Oracle on `val_full`

Run:

```bash
python3 -m scripts.evaluation.evaluate_model \
  evaluation.run_dir=<RUN_DIR> \
  evaluation.model_name=<MODEL_NAME> \
  dataset.active_subsets.val=val_full \
  validation=sliding_window_3d_metrics_full \
  evaluation.threshold_protocol.mode=sweep_with_oracle \
  evaluation.threshold_protocol.thresholds=0.05:0.90:0.05
```

Verify:

- full validation completes
- report compares `0.5`, global best, and oracle
- volume ratio summaries are available

### Success Criteria

The output summary should let the team answer:

- Does best global threshold improve meaningfully over `0.5`?
- Does oracle improve meaningfully over best global?
- Is there a systematic predicted-volume bias?

### Rollback

No rollback needed for validation. If results reveal metric mismatch, pause and investigate inference/metric parity before using outputs scientifically.

---

## 14. Testing Matrix

### Unit Tests

Run:

```bash
python3 -m pytest \
  tests/test_evaluation_threshold_protocol.py \
  tests/test_evaluation_threshold_selection.py \
  tests/test_evaluation_threshold_records.py \
  tests/test_evaluation_model_config.py \
  tests/test_evaluation_model_loader.py \
  tests/test_evaluation_io_model_volumes.py
```

### Integration Tests

Run:

```bash
python3 -m pytest \
  tests/test_evaluation_pipeline.py \
  tests/test_evaluation_model_entrypoint.py \
  tests/test_evaluation_entrypoints_integration.py
```

### Existing Evaluation Regression Tests

Run:

```bash
python3 -m pytest tests/test_evaluation_*.py
```

### Minimal Syntax Checks

Run:

```bash
python3 -m py_compile scripts/evaluation/evaluate_model.py
python3 -m py_compile scripts/evaluation/slurm_runners/run_evaluate_model.py
python3 -m py_compile scripts/evaluation/evaluation_pipeline.py
python3 -m py_compile scripts/evaluation/io_model_volumes.py
python3 -m py_compile scripts/evaluation/threshold_records.py
```

---

## 15. Implementation Order and Dependencies

Recommended order:

1. Phase 0: Baseline audit.
2. Phase 1: Add configs.
3. Phase 2: Add config loading and override utilities.
4. Phase 3: Add model loading utilities.
5. Phase 4: Extend threshold protocol.
6. Phase 5: Add threshold records and oracle aggregation.
7. Phase 6: Add IO producers.
8. Phase 7: Add evaluation pipeline.
9. Phase 8: Add entrypoint.
10. Phase 9: Update docs and legacy guidance.
11. Phase 10: Run ISLES26 3D validation.

Dependency notes:

- Phase 7 depends on Phases 2, 3, 4, 5, and 6.
- Phase 8 depends on Phase 7.
- Phase 10 depends on all runtime phases.
- Documentation can be partially updated earlier, but final examples should wait until the entrypoint stabilizes.

---

## 16. Rollback and Contingencies

### Safe Rollback Strategy

Because the implementation primarily adds new modules and a new entrypoint, rollback is straightforward:

1. Remove new config files under `configs/evaluation/`.
2. Remove new `scripts/evaluation/model_config.py`, `scripts/evaluation/model_loader.py`, and `scripts/evaluation/evaluation_pipeline.py`.
3. Remove `scripts/evaluation/io_model_volumes.py`.
4. Remove `scripts/evaluation/threshold_records.py`.
5. Revert modifications to:
   - `scripts/evaluation/threshold_protocol.py`
   - `scripts/evaluation/contracts.py`
   - `scripts/evaluation/reporting.py`
   - `scripts/evaluation/metrics_engine.py`
   - README files
6. Remove new tests.

Existing nnU-Net evaluation and legacy 2D live-model evaluation should remain usable throughout unless a shared module is modified incorrectly.

### Contingency: Hydra Override Complexity

If pure Hydra composition becomes awkward because evaluation must load a saved run config first:

- Use an argparse-based entrypoint similar to `scripts/nnunet/evaluate_nnunet_results.py`.
- Accept config preset names and trailing `key=value` overrides.
- Internally use OmegaConf to load run config, merge evaluation config, and apply overrides.

This preserves the user-facing config-driven behavior even if the entrypoint is not a pure `@hydra.main` script.

### Contingency: Future 3D Diffusion Support

The new `io_model_volumes.py` module should remain representation-based, not discriminative-specific. Current 3D non-discriminative diffusion is rejected because existing diffusion adapters are 2D-shaped, not because the IO design should permanently exclude diffusion.

Before 3D diffusion volume evaluation is enabled, the diffusion stack should satisfy the contract documented in `src/diffusion/diffusion.py`:

1. `sample(conditioned_image)` accepts 5D conditioning volumes `[B, C, H, W, D]`.
2. `sample(conditioned_image)` returns probability volumes `[B, C_mask, H, W, D]` aligned with the ground-truth target.
3. Forward training paths accept 5D noisy masks, conditioning volumes, and timestep tensors without implicit 2D shape assumptions.
4. Configs define 3D mask shape, ROI, and full-volume versus patch/sliding-window inference policy explicitly.

### Contingency: 3D Metrics Too Slow

If threshold sweeps with full 3D metrics are too expensive:

- Default to a smaller metric list for sweep mode.
- Allow full metrics only at selected thresholds.
- Keep `val_fast` as default for exploratory threshold sweeps.
- Add config option:

```yaml
evaluation:
  threshold_protocol:
    expensive_metrics_at:
      - fixed
      - best_global
      - oracle
```

This is optional and should not be added unless runtime becomes a blocker.

### Contingency: Metric Parity Mismatch

If threshold `0.5` evaluation does not match training validation:

1. Verify the same checkpoint is loaded.
2. Verify final resolved config matches the training validation config.
3. Verify `dataset.active_subsets.val`.
4. Verify direct vs sliding-window inference mode.
5. Verify metric class and thresholding semantics.
6. Add a targeted regression test once the cause is identified.

---

## 17. Deliverables Checklist

### Runtime

- [ ] Config group for model evaluation exists.
- [ ] New model evaluation entrypoint exists.
- [ ] Run config loading and overrides work.
- [ ] Checkpoint loading is reusable.
- [ ] 2D live-model evaluation remains supported.
- [ ] 3D discriminative volume evaluation works.
- [ ] Current 3D non-discriminative diffusion fails explicitly with a capability error.
- [ ] Fixed threshold mode works.
- [ ] Sweep mode works.
- [ ] Oracle mode works.
- [ ] Sweep-with-oracle mode works.

### Reports

- [ ] `canonical_results.json`
- [ ] `evaluation_summary.txt`
- [ ] `resolved_evaluation_config.yaml`
- [ ] per-threshold aggregate CSV
- [ ] per-case/per-threshold CSV
- [ ] oracle per-case CSV
- [ ] threshold `0.5` versus best global versus oracle comparison

### Tests

- [ ] threshold protocol tests
- [ ] threshold record/oracle tests
- [ ] config loading tests
- [ ] model loader tests
- [ ] 3D model-volume IO tests
- [ ] evaluation pipeline tests
- [ ] entrypoint tests
- [ ] existing evaluation regression tests pass

### Documentation

- [ ] `scripts/evaluation/README.md` documents new entrypoint.
- [ ] `scripts/analysis/README.md` marks legacy threshold analysis path.
- [ ] Example commands use `python3 -m ...` notation.

---

## 18. Resolved Planning Decisions

1. Use `<run_dir>/analysis/evaluation_v3/` as the default output root for the new model evaluation path.
2. Focus the first implementation on JSON/CSV/text outputs; defer plots.
3. Target 3D volume evaluation first. Keep the legacy 2D live-model entrypoint untouched until the new path is validated.
4. Organize live-model IO by representation: existing `io_diffusion.py` remains the legacy 2D slice producer, and new `io_model_volumes.py` handles 3D volumes.
5. Reject current 3D non-discriminative diffusion with a capability error because current diffusion adapters are 2D-shaped. Future 3D diffusion should plug into `io_model_volumes.py` once it satisfies the volume contract.
6. Force/include threshold `0.5` when needed for fixed-threshold comparison.

---

## 19. Recommended First Implementation Slice

To reduce risk, implement the first functional slice as:

1. Config loader and model loader utilities.
2. 3D volume producer supporting current discriminative adapters and rejecting current 3D non-discriminative diffusion with a capability error.
3. Per-case threshold records for volume metrics.
4. `fixed` and `sweep_with_oracle` protocol for volume level.
5. New entrypoint.
6. Tests for the above.

Defer:

- plotting
- 2D entrypoint migration
- compatibility wrapper changes
- full deprecation of `scripts/analysis/threshold_analysis.py`

This first slice directly addresses the current ISLES26 threshold-calibration question while keeping changes scoped.

