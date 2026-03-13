# PRD Handover: ISLES24 Explicit Data Contract Refactor

Date: 2026-03-12  
Audience: Local development continuation, then cluster rollout  
Scope: PRD "Explicit Data Contract Refactor for ISLES24 (Config + Loader Routing + Script Alignment)"

---

## 1) Executive State

The refactor is in a late implementation state:

- Core config contract is in place and being consumed.
- Loader routing is explicit via `data_mode.loader_mode`.
- A new nnUNet 2D training backend (`ISLES24NNUNet2D`) is implemented.
- Key scripts now validate canonical config preconditions.
- Runtime smoke validation is still pending on full local/cluster environments.

Current completion level against PRD: approximately late Phase 4 / early Phase 5.

---

## 2) PRD Contract Evolution and Final Design Choices

This section captures how the implementation evolved from the original PRD/CAP wording into the current, chosen contract.

### A. Runtime profile taxonomy changed from initial PRD wording

Initial PRD examples referenced runtime files like:

- `cluster_heavy_io.yaml`
- `cluster_balanced_io.yaml`
- `local_debug_io.yaml`

Final adopted runtime configs are:

- `configs/data_runtime/cluster.yaml`
- `configs/data_runtime/ddp_cluster.yaml`
- `configs/data_runtime/local.yaml`

Rationale:

- Better alignment with existing environment semantics and job naming.
- Cleaner top-level override usage for DDP runs.

### B. `data_profile` semantics were intentionally narrowed

Initial CAP considered profile composites as a place where runtime could be bundled.

Final decision:

- `data_profile` is semantic/composition only (`dataset + data_mode + data_io`).
- Runtime is selected independently via `data_runtime` at top-level.

Rationale:

- Prevents hidden performance/runtime changes when switching data profiles.
- Keeps experiment intent explicit.

### C. nnUNet dataset identity ownership moved from `data_io` to `dataset`

Initial PRD examples implied nnUNet identity fields in IO config variants.

Final decision:

- nnUNet identity is owned by dataset profile:
  - `dataset.nnunet.dataset_id`
  - `dataset.nnunet.dataset_name`
- `data_io` now focuses on location/path source.

Rationale:

- Dataset identity is a semantic characteristic, not an IO transport concern.
- Avoids duplicating near-identical nnUNet IO files per modality profile.

### D. nnUNet IO configs were deduplicated

Initial PRD file inventory suggested separate files like:

- `isles24_nnunet_050_baseline.yaml`
- `isles24_nnunet_051_team1.yaml`
- `isles24_nnunet_052_*`

Final decision:

- One canonical nnUNet 2D IO file is used:
  - `configs/data_io/isles24_nnunet_2d.yaml`
- Profile-specific dataset id/name is resolved from `dataset.nnunet.*`.

Rationale:

- Less config sprawl, fewer sync errors, easier maintenance.

### E. Environment group was slimmed to runtime context only

Final environment policy:

- `environment.dataset` supplies dataset path context (`data_root`, `split_file`, `nnunet_root`).
- Loader/runtime knobs are not sourced from environment anymore (now in `data_runtime`).

Rationale:

- Removes old environment-as-contract aliasing behavior.
- Enforces one canonical location for runtime knobs.

### F. Legacy compatibility bridges were intentionally not retained

Final policy aligns with PRD migration guidance:

- No long-lived compatibility shim for old key paths.
- Scripts and entrypoints moved to canonical keys + fail-fast checks.

Rationale:

- Reduces long-term ambiguity and maintenance debt.

### G. Loader naming and routing outcomes

Final state:

- Routing source: `data_mode.loader_mode` only.
- Online 2D path retains current implementation and explicit alias:
  - `ISLES24OnlineProc2D = ISLES24Dataset2D`
- New backend implemented:
  - `ISLES24NNUNet2D`

Rationale:

- Meets explicit dispatch requirement while minimizing disruptive renames.

### H. DDP environment choice simplified

Earlier handover notes mentioned preserving a thin DDP environment shim.

Current final choice in active configs:

- DDP configs override environment to `isles_cluster`.
- DDP behavior is represented via `distribution=ddp` and `data_runtime=ddp_cluster`.

Rationale:

- Removes dependency on a separate environment alias and avoids missing-target errors.

---

## 3) What Has Been Concretely Achieved

## A. Config Contract and Composition

The codebase now uses explicit data groups:

- `configs/dataset/*`
- `configs/data_mode/*`
- `configs/data_io/*`
- `configs/data_runtime/*`
- `configs/data_profile/*`

Top-level configs compose these groups explicitly (instead of implicit dataset-only routing), including nnUNet variants such as:

- `configs/local_nnunet2d_baseline.yaml`
- `configs/cluster_nnunet2d_baseline.yaml`

## B. Loader Contract Adoption

`src/data/loaders.py` now:

- Validates the explicit contract in `validate_dataset_contract(cfg)`.
- Dispatches by `cfg.data_mode.loader_mode`.
- Supports:
  - `online_slices_3d_to_2d`
  - `full_volumes_3d`
  - `nnunet_slices_2d` (new backend implemented)

## C. nnUNet 2D Backend Added

Implemented `ISLES24NNUNet2D` in `src/data/loaders.py` with:

- nnUNet folder contract checks:
  - `Dataset{dataset_id}_{dataset_name}`
  - `imagesTr/labelsTr` and `imagesTs/labelsTs`
- Channel completeness checks (`_0000 ... _{C-1:04d}` per sample)
- Geometry consistency checks (channel to channel and image to label)
- Identity normalization for downstream compatibility:
  - `"<volume_id>_slice<idx>"`

## D. Caching Behavior Improvements

- New nnUNet loader cache implemented.
- Existing online 2D loader cache corrected to avoid train/val collisions by using split-scoped keys:
  - `("tr", case_idx)` and `("ts", case_idx)` style behavior.

## E. Script Contract Hardening (Key/Preconditions)

Updated scripts to validate canonical config keys and preconditions:

- `scripts/nnunet/convert_isles24_2d_dataset_to_nnunet.py`
  - Added converter-specific precondition validation.
  - Enforces expected `loader_mode`, `dim`, and required key presence.
- `scripts/evaluation/compute_segmentation_metrics_for_diffusionmodel_2d_predictions.py`
  - Added validation for 2D slice-based config preconditions.
- `scripts/analysis/threshold_analysis.py`
  - Added validation for 2D slice-based config preconditions.

## F. DDP Config Reference Issue Resolved

The missing `isles_ddp_cluster` reference issue has been removed from active config usage.

---

## 4) Changes Made in This Laptop Session

Files edited:

- `src/data/loaders.py`
  - Added `ISLES24NNUNet2D`
  - Implemented `nnunet_slices_2d` dispatch
  - Updated caching keys for online 2D to be split-safe
- `scripts/nnunet/convert_isles24_2d_dataset_to_nnunet.py`
  - Added `validate_converter_contract(cfg)`
  - Canonical key checks and fail-fast preconditions
- `scripts/evaluation/compute_segmentation_metrics_for_diffusionmodel_2d_predictions.py`
  - Added `validate_eval_config_contract(cfg)`
- `scripts/analysis/threshold_analysis.py`
  - Added `validate_threshold_analysis_config_contract(cfg)`

---

## 5) PRD Acceptance Criteria Status

1. Three loader modes represented + validated: **Implemented in code**  
2. Top-level configs compose explicit groups: **Implemented**  
3. `get_dataloaders` dispatches via `data_mode.loader_mode`: **Implemented**  
4. Profiles/naming conventions present: **Implemented**  
5. Scripts updated and passing smoke execution: **Partially complete (updates done, smoke pending)**  
6. Documentation/examples reflect final contract: **Partially complete**

---

## 6) Known Unverified Items (Important)

These are not failures, but pending runtime confirmation:

- `ISLES24NNUNet2D` has not yet been smoke-tested in target environment.
- No full compose/runtime matrix has been executed in this laptop session.
- No performance/timing baseline comparison run yet for nnUNet mode.

Keep these explicit in handover discussions and in PR closeout notes.

---

## 7) Remaining Work to Bring the PRD to Completion

## Phase 5 Completion (recommended order)

1. Run script-level smoke checks on local environment.
2. Confirm identity and metadata assumptions in end-to-end evaluation flow.
3. Final pass for any stale config-key assumptions outside already-touched scripts.

## Phase 6 Closeout

1. Finalize docs/README examples with canonical key usage.
2. Decide deprecation/removal status for any remaining legacy config artifacts.
3. Execute final mode matrix smoke checks and record evidence.

---

## 8) Local Environment Execution Plan (Concrete)

Use this exact order to reduce debugging ambiguity.

1) Config resolve checks (no training):

- `python start_training.py --config-name local --cfg job`
- `python start_training.py --config-name local_nnunet2d_baseline --cfg job`
- `python start_training.py --config-name cluster --cfg job`
- `python start_training.py --config-name cluster_nnunet2d_baseline --cfg job`

2) Dataloader-level smoke (short):

- `python scripts/test_validation_memory.py --config-name local`
- `python scripts/test_validation_memory.py --config-name local_nnunet2d_baseline`

Recommended overrides for smoke speed/stability:

- `data_runtime.num_train_workers=0`
- `data_runtime.num_valid_workers=0`
- `data_runtime.num_test_workers=0`
- `validation.val_batch_size=1`

3) Converter smoke:

- `python -m scripts.nnunet.convert_isles24_2d_dataset_to_nnunet --config-name convert_nnunet_local nnunet.test=true nnunet.test_max_slices=20`

4) Minimal training smoke for nnUNet mode:

- `python start_training.py --config-name local_nnunet2d_baseline training.max_steps=5 data_runtime.num_train_workers=0 data_runtime.num_valid_workers=0 data_runtime.num_test_workers=0`

5) Evaluation script smoke (using an existing run dir/checkpoint):

- `python -m scripts.evaluation.compute_segmentation_metrics_for_diffusionmodel_2d_predictions --run-dir <run_dir> --model-name <checkpoint_name> --test --test-max-slices 10`
- `python scripts/analysis/threshold_analysis.py --run-dir <run_dir> --model-name <checkpoint_name>`

---

## 9) Cluster Rollout Plan (After Local Green)

1. Re-run config resolves for cluster targets:

- `cluster`
- `cluster_1M_ddp`
- `cluster_500K_64b_4gpu_ddp`
- `cluster_nnunet2d_baseline`

2. Run one short cluster smoke job for:

- Online 2D path
- nnUNet 2D path

3. Monitor:

- Dataloader startup time
- Worker stability
- Batch tensor shape/contract
- sample-id formatting (`<volume_id>_slice<idx>`)

4. Then run intended full jobs.

---

## 10) Deferred/Non-Blocking Notes

These are intentionally deferred to keep the task scoped:

- `ISLES24Dataset2D` metadata export support is still minimal; converter currently falls back safely when metadata tuple is absent.
- Caching in DDP is process-local/worker-local, not a global shared cache across ranks (expected behavior with standard PyTorch DataLoader process model).

---

## 11) Suggested Exit Criteria for Final PRD Sign-Off

Sign off only when all are true:

- All target configs resolve successfully in local and cluster contexts.
- Online 2D and nnUNet 2D modes both pass smoke runs.
- Converter smoke run succeeds under canonical config contract.
- Diffusion evaluation and threshold analysis scripts run without config-key errors.
- Final docs include canonical key examples and mode selection guidance.

---

## 12) Quick Handover Summary

You are not blocked on architecture anymore; remaining work is operational validation and final polish.

The safest next move is:

1) local matrix smoke -> 2) local short run timing -> 3) cluster smoke -> 4) finalize docs/cleanup.

