# CAP: Change Action Plan for ISLES24 Explicit Data Contract Refactor

Version: 1.0  
Date: 2026-03-10  
Status: Ready for execution  
Depends on: `docs/PRD_ISLES24_Explicit_Data_Contract_Refactor.md`

## 1. Overall Plan Summary

Execution will proceed in ordered phases to reduce risk and maintain clear rollback boundaries:

1. Baseline and guardrails.
2. Config contract scaffolding.
3. Core loader routing and validation refactor.
4. Experiment config migration.
5. nnUNet 2D loader backend integration.
6. Post-transition script alignment.
7. Cleanup and final validation.

Key policy:

- No during-change compatibility layer for scripts.
- Scripts are aligned after core source and config contract transition.

## 2. Phase Breakdown

## Phase 0: Baseline Snapshot and Guardrails

### Goal

Capture reference behavior before refactor.

### Tasks

1. Record baseline loader outputs and sample id format.
2. Run short smoke pass for current online 2D path.
3. Record key invariants:
   - output tensor shapes
   - sample id format
   - modality count agreement
   - dataloader keys (`train`, `val`, `sample`)

### Validation

- Baseline notes captured and reproducible.

### Rollback

- No source change in this phase.

## Phase 1: Config Contract Scaffolding

### Goal

Add explicit data config groups without changing runtime behavior.

### Tasks

1. Create `configs/data_mode/*`.
2. Create `configs/dataset/isles24_base.yaml` and modality profile files.
3. Create `configs/data_io/*` including:
   - `isles24_online_volumes.yaml`
   - `isles24_nnunet_{dataset_id}_{idkey}.yaml` variants
4. Create `configs/data_runtime/*` preserving existing knobs.
5. Add `configs/data_profile/*` composites for:
   - online 2D baseline
   - nnUNet 2D baseline
   - full-volume 3D baseline

### Files

- New files under:
  - `configs/data_mode/`
  - `configs/dataset/`
  - `configs/data_io/`
  - `configs/data_runtime/`
  - `configs/data_profile/`

### Validation

- Hydra resolve passes for each profile.
- No behavior change in existing training path.

### Rollback

- Remove new config files only.

## Phase 2: Core Loader Contract Refactor

### Goal

Move loader selection to explicit mode-based dispatch.

### Tasks

1. Add contract validator in loader path.
2. Replace `dataset.name` route logic with `data_mode.loader_mode`.
3. Ensure output contract remains stable.
4. Preserve or normalize sample id format expected by evaluation.

### Files

- `src/data/loaders.py` (primary)
- Optional helper module if extracted for validation.

### Validation

- Smoke test for mode resolution:
  - `online_slices_3d_to_2d`
  - `full_volumes_3d`
- Lint check after edits.

### Rollback

- Revert loader refactor file(s) to pre-phase state.

## Phase 3: Experiment Config Migration

### Goal

Adopt explicit group composition in top-level experiment configs.

### Tasks

1. Update cluster and local top-level configs to compose:
   - `/dataset`
   - `/data_mode`
   - `/data_io`
   - `/data_runtime`
2. Keep online 2D as default-safe migration path.
3. Add at least one nnUNet-composed experiment config.

### Files

- `configs/cluster*.yaml`
- `configs/local*.yaml` where applicable

### Validation

- Compose checks pass.
- Existing online training still launches.

### Rollback

- Revert top-level config composition changes.

## Phase 4: nnUNet 2D Loader Integration

### Goal

Provide `nnunet_slices_2d` backend under canonical contract.

### Tasks

1. Implement nnUNet 2D dataset class.
2. Parse channel files and labels per nnUNet naming.
3. Enforce channel completeness and shape consistency.
4. Integrate with dataloader factory route.
5. Normalize sample id for downstream compatibility where needed.

### Files

- `src/data/loaders.py` or a dedicated module under `src/data/`.

### Validation

- Loader smoke run on nnUNet profile.
- Batch shapes and sample IDs verified.
- Lint check.

### Rollback

- Disable route or revert nnUNet loader integration changes.

## Phase 5: Post-Transition Script Alignment

### Goal

Align script contract usage to canonical keys after core migration.

### Tasks

1. Update converter script to canonical groups and mode assertions.
2. Update diffusion/custom evaluation scripts to canonical contract.
3. Update analysis script contract checks.
4. Update SLURM runner help and override examples.
5. Remove legacy alias mutation patterns in scripts.

### Files

- `scripts/nnunet/convert_isles24_2d_dataset_to_nnunet.py`
- `scripts/evaluation/compute_segmentation_metrics_for_diffusionmodel_2d_predictions.py`
- `scripts/analysis/threshold_analysis.py`
- `scripts/evaluation/io_diffusion.py` (if identity normalization needed)
- related runner scripts for examples/help text.

### Validation

- Script smoke runs after transition:
  - converter dry/small run
  - diffusion evaluation test mode
  - threshold analysis with existing run dir

### Rollback

- Revert script updates as one phase batch.

## Phase 6: Cleanup and Finalization

### Goal

Finalize transition and retire legacy assumptions.

### Tasks

1. Mark legacy configs deprecated or remove after decision.
2. Update docs and examples to canonical contract only.
3. Execute final mode matrix smoke checks.
4. Record final acceptance evidence.

### Validation

- All acceptance criteria in PRD met.

### Rollback

- Restore deprecated configs if needed while keeping canonical path intact.

## 3. Validation Matrix per Phase

| Phase | Validation Target | Pass Condition |
|---|---|---|
| 0 | Baseline invariants | Shapes/IDs documented |
| 1 | Config resolve | All new profiles resolve |
| 2 | Loader routing | Mode dispatch works; online path intact |
| 3 | Experiment composition | Updated configs launch with explicit groups |
| 4 | nnUNet loader | nnUNet profile produces valid batches |
| 5 | Scripts | Updated scripts run with canonical keys |
| 6 | End-to-end | Final matrix passes and docs updated |

## 4. Risk Handling and Contingencies

1. If loader contract breaks training:
   - revert Phase 2 file set and restore online baseline profile.
2. If nnUNet integration fails:
   - keep canonical contract but disable `nnunet_slices_2d` route temporarily.
3. If scripts fail after core transition:
   - pause rollout, complete Phase 5 before any operational usage.
4. If identity mismatch appears:
   - enforce single normalization location (loader or eval ingress).

## 5. Operational Execution Order

Strict order:

1. Phase 0  
2. Phase 1  
3. Phase 2  
4. Phase 3  
5. Phase 4  
6. Phase 5  
7. Phase 6

No phase should begin without previous phase validation pass.

## 6. Deliverables

By completion, expected deliverables:

1. Canonical config groups and naming conventions in repository.
2. Mode-based loader routing and validator in core source.
3. nnUNet 2D loader backend under canonical contract.
4. Updated top-level experiment composition.
5. Post-transition script alignment to canonical keys.
6. Finalized docs and deprecation notes.

