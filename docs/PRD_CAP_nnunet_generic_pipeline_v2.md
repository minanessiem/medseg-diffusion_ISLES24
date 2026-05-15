## PRD v2: Generic nnU-Net Pipeline (Config-Driven, Cluster-Compatible)

**Status:** Draft for review  
**Scope:** nnU-Net conversion, command execution, and evaluation workflow unification  
**Primary intent:** Replace dataset/dim-specific scripts with generic entrypoints that compose existing config groups and preserve SLURM login-node constraints.

---

## 1) Overview and Task Description

The repository currently has working nnU-Net workflows, but they are split into scripts that encode assumptions like ISLES24, 2D slices, and specific naming conventions. This PRD proposes a generic nnU-Net workflow with:

- one conversion entrypoint,
- one command runner entrypoint,
- one evaluation entrypoint,

while keeping separate SLURM runner scripts for cluster submission from login nodes.

The design is intentionally incremental: no broad package cleanup, no relocation of shared SLURM utility code in this PRD, and backward compatibility with current commands during migration.

---

## 2) Requirements and Scope

### Functional requirements

- Conversion must support both:
  - `online_slices_3d_to_2d` export to nnU-Net 2D dataset format,
  - `full_volumes_3d` export to nnU-Net 3D dataset format.
- nnU-Net command execution must support `preprocess`, `train`, and `predict`.
- Evaluation must support:
  - slice-form nnU-Net outputs (`*_sXXXX` identity),
  - native volume-form nnU-Net outputs (one prediction volume per case).
- All workflows must be driven by config composition and CLI overrides, not dataset-specific scripts.

### Non-functional requirements

- Preserve cluster usability from login node (no heavy runtime deps in runner scripts).
- Reuse existing SLURM helper patterns (`commandline_utils`, `load_config` in `single_job_runner`).
- Preserve existing behavior for ISLES24 2D baseline during migration.
- Keep code modular and strategy-based to avoid copy-paste forks.

### Out of scope

- Refactoring old shared utility module locations.
- Rewriting metric implementations in `src/metrics/metrics.py`.
- Changing training pipeline architecture outside nnU-Net workflow integration.

---

## 3) Current-State Constraints (Validated)

- Converter currently depends on `get_dataloaders(cfg)` and `validate_dataset_contract(cfg)`, so conversion composition needs more than `data_profile` + `environment` alone.
- Converter currently relies on `cfg.nnunet.*` conversion knobs (`output_dir`, `test`, `parallel`, split export toggles).
- Evaluation I/O for nnU-Net currently assumes slice identity parsing and slice streaming semantics.
- SLURM runner scripts must remain lightweight due login-node package limitations.

---

## 4) Final Config Strategy (Updated)

### 4.1 Source of truth

- Local vs cluster concerns must live in `configs/environment/*`.
- Dataset/dimensionality concerns remain in `configs/data_profile/*`, `configs/data_mode/*`, `configs/data_io/*`, `configs/dataset/*`.
- Runtime loader requirements remain in existing `data_runtime`, `validation`, and `model` groups as currently required by converter internals.

### 4.2 Organized nnU-Net workflow presets

Introduce organized preset paths under `configs/nnunet/`:

- `configs/nnunet/convert/*`
- `configs/nnunet/eval/*`
- `configs/nnunet/run/*`

This is for structure and discoverability, not for duplicating environment concerns.

### 4.3 No local/cluster split inside `nnunet_run`

`nnunet_run` should be environment-agnostic and minimal.  
If any local/cluster differences exist, they must be inherited from `environment` and SLURM defaults, not duplicated in `nnunet_run/local.yaml` vs `cluster.yaml`.

### 4.4 Backward compatibility during migration

Existing top-level conversion presets like `configs/convert_nnunet_*.yaml` can be retained as aliases/wrappers initially, then deprecated after adoption.

---

## 5) Components of Interest

- Conversion script: `scripts/nnunet/convert_isles24_2d_dataset_to_nnunet.py`
- Existing nnU-Net command runner: `scripts/nnunet/slurm_runners/run_nnunet_command.py`
- Existing nnU-Net eval scripts:
  - `scripts/nnunet/compute_segmentation_metrics_for_nnunet_2d_predictions.py`
  - `scripts/evaluation/compute_segmentation_metrics_for_nnunet_2d_predictions.py`
- nnU-Net slice I/O: `scripts/evaluation/io_nnunet.py`
- Evaluation engine and 3D metrics registry:
  - `scripts/evaluation/metrics_engine.py`
  - `scripts/evaluation/metrics_registry_3d.py`
- SLURM utility reuse points:
  - `scripts/slurm/utils/commandline_utils.py`
  - `scripts/slurm/single_job_runner.py`
  - `scripts/slurm/base_run_config.py`
  - `scripts/slurm/job_runner.py`

---

## 6) Necessary Changes

### 6.1 Generic conversion entrypoint

Create `scripts/nnunet/convert_to_nnunet.py` with strategy-based export routing:

- `SliceExportStrategy` for `online_slices_3d_to_2d`.
- `VolumeExportStrategy` for `full_volumes_3d`.

Core behavior remains config-driven and uses current loader contracts.  
The old conversion script becomes a compatibility wrapper.

### 6.2 Generic evaluation entrypoint

Create `scripts/nnunet/evaluate_nnunet_results.py` with I/O adapters:

- slice adapter for current nnU-Net 2D naming/streaming,
- volume adapter for case-level 3D prediction/GT pairing.

Reuse existing reporting and metrics engine to keep output schema stable.

### 6.3 Command runner unification

Evolve `scripts/nnunet/slurm_runners/run_nnunet_command.py` to optionally resolve defaults from composed config while preserving explicit CLI override precedence.

### 6.4 SLURM runner additions

Create generic lightweight runners:

- `scripts/nnunet/slurm_runners/run_convert_to_nnunet.py`
- `scripts/nnunet/slurm_runners/run_evaluate_nnunet_results.py`

Both must reuse existing helper patterns and avoid heavy package imports.

### 6.5 Config reorganization

Add organized workflow preset files under `configs/nnunet/{convert,eval,run}` and migrate command docs/examples accordingly.

---

## 7) `nnunet_eval` Key Justification

Minimal dedicated eval policy keys are justified even with existing data configs:

- `nnunet_eval.input_format`: required to choose slice-adapter vs volume-adapter semantics.
- `nnunet_eval.levels`: controls whether to compute slice metrics, volume metrics, or both.
- `nnunet_eval.threshold`: explicit fixed-threshold policy and reproducible report metadata.
- `nnunet_eval.allow_shape_mismatch`: operational resilience for large batch evaluations where strict failure may be undesirable.

These keys are workflow-policy knobs, not dataset/environment knobs, so they belong in eval-specific config.

---

## 8) `nnunet_run` Key Set (Minimal)

`nnunet_run` remains small and environment-agnostic. Candidate keys:

- `default_fold` (e.g., `all`)
- `default_checkpoint` (e.g., `checkpoint_final.pth`)
- `save_probabilities` default
- command-specific fallback process counts only if not inferred by CLI/config
- optional mapping from dimension to nnU-Net configuration (`2d`, `3d_fullres`)

No path roots, no cluster/local resource duplication in this group.

---

## 9) Expected Output State

After implementation:

- Generic runtime scripts exist for convert/run/eval.
- Legacy scripts still work as wrappers with deprecation warnings.
- Configs are more orderly under `configs/nnunet/{convert,eval,run}`.
- Environment remains the single source of local/cluster differences.
- ISLES24 2D and ISLES26 3D nnU-Net workflows both operate from config composition.

---

## 10) Risks and Mitigations

- Risk: regression in current ISLES24 2D baseline behavior.
- Mitigation: strict parity checks for conversion outputs and eval summaries before deprecating old commands.

- Risk: ambiguity in auto-detecting eval input format.
- Mitigation: explicit `input_format` override plus conservative `auto` behavior with clear errors.

- Risk: SLURM runner drift and duplicated config parsing logic.
- Mitigation: enforce reuse of `commandline_utils` and `single_job_runner.load_config`.

- Risk: config sprawl during reorganization.
- Mitigation: keep workflow groups thin; avoid duplicating values already in dataset/environment/runtime groups.

---

## 11) Assumptions and Dependencies

- nnU-Net v2 commands are available inside target execution environments.
- Existing dataset contracts for ISLES24 and ISLES26 remain valid.
- SLURM submission flow (`sbatch` via `SlurmJobRunner`) remains unchanged.
- Existing evaluation engine and reporting schema remain canonical output format.

---

## 12) Acceptance Criteria

- A user can run conversion/evaluation generically using `--config-name` presets under `configs/nnunet/...` without selecting dataset-specific script files.
- Local/cluster differences are resolved via `environment` composition, not duplicated `nnunet_run` local/cluster files.
- Login-node runners remain dependency-light and reuse existing shared config/CLI helper code.
- Legacy commands remain functional during migration window.

## CAP v2: Generic nnU-Net Pipeline (Config-First, Cluster-Safe)

### Overall Plan Summary
- Implement a generic nnU-Net workflow with three entrypoints: `convert`, `run`, `evaluate`.
- Keep strict separation between runtime scripts and login-node `slurm_runners`.
- Reuse existing config ecosystem (`data_profile`, `data_mode`, `data_io`, `environment`, `data_runtime`, `validation`, `model`, `nnunet`) as primary sources of truth.
- Reorganize config files under `configs/nnunet/{convert,eval,run}` while preserving old entry aliases during migration.
- Avoid package-structure refactors; reuse `scripts/slurm/utils/commandline_utils.py` and `scripts/slurm/single_job_runner.py` directly.

---

## Phase 0 — Baseline Lock and Regression Contracts
- Record current behavior for `scripts/nnunet/convert_isles24_2d_dataset_to_nnunet.py`, `scripts/nnunet/slurm_runners/run_nnunet_command.py`, and current nnU-Net eval scripts.
- Capture expected artifacts: folder layout, naming conventions, `dataset.json` keys, metrics report files, SLURM dry-run outputs.
- Create a baseline document at `docs/nnunet_baseline_parity_contract.md`.
- Validation: parity contract is explicit and executable as checklist.
- Rollback: none (docs only).

## Phase 1 — Config Reorganization (No Functional Change)
- Create `configs/nnunet/convert/*` and move current convert presets there.
- Create `configs/nnunet/eval/*` with thin policy defaults (`input_format`, `levels`, `threshold`, `allow_shape_mismatch`).
- Create optional `configs/nnunet/run/default.yaml` for nnU-Net command semantics only; no `local`/`cluster` variants here.
- Keep environment differences in `configs/environment/*`; optionally rename `isles24_nnunet2d_cluster` -> `isles24_nnunet_cluster` with compatibility alias.
- Add top-level compatibility alias configs (`configs/convert_nnunet_*.yaml`) that default-chain into new `configs/nnunet/convert/*`.
- Validation: both old and new `--config-name` paths compose successfully.
- Rollback: keep top-level aliases as source of truth until migration is stable.

## Phase 2 — Conversion Core Extraction (2D Parity First)
- Introduce `scripts/nnunet/core/contracts.py`, `scripts/nnunet/core/config_resolvers.py`, `scripts/nnunet/core/exporters.py`, `scripts/nnunet/core/conversion_pipeline.py`.
- Move current slice export logic into `SliceExportStrategy` without changing external behavior.
- Keep existing converter script functional, now delegating to extracted core.
- Preserve current use of existing data stack (`get_dataloaders`, `validate_dataset_contract`).
- Validation: ISLES24 2D output parity against Phase 0 contract.
- Rollback: retain legacy inlined path behind a temporary flag if parity fails.

## Phase 3 — 3D Volume Export Strategy
- Add `VolumeExportStrategy` in `scripts/nnunet/core/exporters.py` for `full_volumes_3d`.
- Implement strategy resolution from existing config contracts (`data_mode.loader_mode`, `data_mode.dim`).
- Extend provenance output to support volume-level export manifests.
- Generalize `dataset.json` construction for both slice and volume export modes.
- Validation: ISLES26 3D conversion produces valid nnU-Net raw dataset structure and counts.
- Rollback: keep volume strategy opt-in until validation matrix is green.

## Phase 4 — Generic Conversion Entrypoint and Compatibility Wrapper
- Add `scripts/nnunet/convert_to_nnunet.py` as the canonical conversion entrypoint.
- Convert `scripts/nnunet/convert_isles24_2d_dataset_to_nnunet.py` to thin compatibility wrapper with deprecation warning.
- Ensure wrapper passes through Hydra overrides transparently.
- Validation: old and new commands generate equivalent ISLES24 2D artifacts.
- Rollback: maintain old converter as primary command target temporarily.

## Phase 5 — Generic nnU-Net Evaluation Entrypoint
- Add `scripts/nnunet/evaluate_nnunet_results.py` as canonical evaluator.
- Add `scripts/nnunet/core/io_adapters.py` with `slices_2d` adapter (existing behavior) and `volumes_3d` adapter (new direct volume pairing).
- Add `scripts/nnunet/core/evaluation_pipeline.py` using existing evaluation engine/reporting modules.
- Keep `scripts/nnunet/compute_segmentation_metrics_for_nnunet_2d_predictions.py` as compatibility wrapper.
- Validation: ISLES24 2D report parity and successful ISLES26 3D volume-level metric generation.
- Rollback: wrapper can call old evaluation-v2 script path until new pipeline is certified.

## Phase 6 — Generic SLURM Runners (Login Node Safe)
- Add `scripts/nnunet/slurm_runners/run_convert_to_nnunet.py` and `scripts/nnunet/slurm_runners/run_evaluate_nnunet_results.py`.
- Reuse `scripts/slurm/utils/commandline_utils.py` for config override flags and update behavior.
- Reuse `load_config` from `scripts/slurm/single_job_runner.py` for login-node config composition and metadata resolution.
- Keep runner imports dependency-light (stdlib + existing slurm helpers only).
- Convert old nnU-Net convert/eval runner scripts into thin wrappers.
- Validation: `--dry-run` emits correct sbatch configuration and command strings for old and new paths.
- Rollback: retain old runners as callable fallback entrypoints.

## Phase 7 — Evolve nnU-Net Command Runner
- Keep `scripts/nnunet/slurm_runners/run_nnunet_command.py` as canonical run runner.
- Add optional config-based inference for defaults (`dataset_id`, default configuration from dimension) while preserving explicit CLI priority.
- Keep environment and resource concerns anchored to existing environment/slurm config model.
- Validation: preprocess/train/predict dry-run matrix passes for 2D and 3D use-cases.
- Rollback: disable auto-infer behavior behind flag if ambiguity appears.

## Phase 8 — Tests and Verification Harness
- Add conversion tests: strategy resolution, slice parity, volume export sanity.
- Add evaluation tests: slice adapter, volume adapter, report schema stability.
- Add SLURM runner tests: command construction and dry-run config rendering.
- Add integration smoke scripts under `scripts/nnunet/tests_smoke/` for local/cluster command recipes.
- Validation: all new tests pass and baseline parity checklist is satisfied.
- Rollback: mark unstable tests as expected-fail only if blocker is external (cluster path/runtime).

## Phase 9 — Documentation and Migration Completion
- Add `docs/nnunet_generic_workflow.md` with end-to-end local/cluster examples.
- Add command migration mapping old -> new for convert/eval/runners.
- Add deprecation timeline and removal policy for legacy wrappers.
- Finalize updated PRD/CAP references in docs index.
- Validation: docs align with actual CLI and config paths.
- Rollback: keep wrappers and aliases until one full cycle of stable use.

---

## Implementation Order
1. Phase 0  
2. Phase 1  
3. Phase 2  
4. Phase 3  
5. Phase 4  
6. Phase 5  
7. Phase 6  
8. Phase 7  
9. Phase 8  
10. Phase 9  

This ordering minimizes risk by preserving current behavior while adding generic capability incrementally.

---

## Testing and Validation Matrix
- ISLES24 2D conversion old vs new parity.
- ISLES24 2D evaluation old vs new parity.
- ISLES26 3D conversion validity and completeness.
- ISLES26 3D evaluation volume metrics path.
- SLURM dry-run parity for convert/run/eval old and new runners.
- Config-path reachability for both top-level aliases and nested `configs/nnunet/*` names.

---

## Rollback and Contingencies
- Keep legacy scripts as wrappers until all parity checks pass.
- Keep top-level config aliases (`convert_nnunet_*`) during migration.
- Gate auto-inference behavior in runners with explicit flags.
- Prefer additive changes; avoid deleting old paths in this CAP.
- If cluster runner fails due to composition edge cases, temporarily pin runner to explicit CLI mode while fixing config resolver logic.
s
