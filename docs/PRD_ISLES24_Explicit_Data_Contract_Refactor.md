# PRD: Explicit Data Contract Refactor for ISLES24

Version: 1.0  
Date: 2026-03-10  
Status: Approved  
Owner: Data and Training Pipeline Refactor

## 1. Overview and Task Description

This PRD defines a full refactor of data configuration contracts to remove implicit data-loading behavior and replace it with explicit, composable contracts for:

- `online_slices_3d_to_2d`
- `nnunet_slices_2d`
- `full_volumes_3d`

Current behavior is implicitly coupled to dataset naming and legacy config aliases. The target state separates dataset semantics, loading mode, physical source layout, and runtime I/O knobs into clear Hydra config groups.

This PRD also includes downstream implications for `scripts/nnunet`, `scripts/evaluation`, and `scripts/analysis`.

## 2. Requirements and Scope

### 2.1 Functional Requirements

1. Introduce explicit routing by `data_mode.loader_mode` with canonical values:
   - `online_slices_3d_to_2d`
   - `nnunet_slices_2d`
   - `full_volumes_3d`
2. Split data concerns into config groups:
   - `dataset`
   - `data_mode`
   - `data_io`
   - `data_runtime`
   - optional `data_profile` composites
3. Support naming for nnUNet data IO configs:
   - `isles24_nnunet_{dataset_id}_{idkey}.yaml`
4. Provide composable profiles for:
   - 2D online slicing
   - 2D nnUNet slices
   - 3D full volumes
5. Preserve existing runtime knobs (batch sizes, workers, cache, prefetch, pinning).
6. Add strict validation for mode-specific required keys and invariants.

### 2.2 Non-Functional Requirements

- Explicit, readable config intent.
- Deterministic fail-fast validation.
- Minimal ambiguity in naming.
- Maintainable extension path for future datasets and 3D training.

### 2.3 Out of Scope

- Diffusion algorithm redesign.
- nnUNet format redesign.
- Performance optimization work beyond structural contract changes.

## 3. Problem Statement

The current codebase uses implicit assumptions:

- Dataset name routing determines loader behavior.
- `isles24` effectively means online 2D slicing from volumes.
- Storage format and interface contract are not independently represented.
- Script logic includes legacy aliasing and old key expectations.

This increases integration risk for precomputed nnUNet slice loading and future 3D paths.

## 4. Target Contract Model

## 4.1 Contract Layers

- Descriptive (data at rest): where data is and how it is organized.
- Prescriptive (interface contract): what loader output is promised to training/evaluation.

## 4.2 Canonical Schema

### `dataset` group

- `id`
- `name`
- `fold`
- `modalities` (ordered)
- `num_modalities`
- `label_spec`

### `data_mode` group

- `loader_mode`
- `dim`
- `preprocessing_mode` (descriptive only)

### `data_io` group

- `paths.data_root`
- `paths.split_file`
- `paths.nnunet_root`
- `paths.nnunet_dataset_id`
- `paths.nnunet_dataset_name`

### `data_runtime` group

- `train_batch_size`
- `test_batch_size`
- `num_train_workers`
- `num_valid_workers`
- `num_test_workers`
- `use_caching`
- `use_shared_cache`
- `train_prefetch_factor`
- `test_prefetch_factor`
- `persistent_workers.train|val|test`
- `pin_memory.train|val|test`

## 5. Config Structure and Naming

## 5.1 Dataset identity and modalities

- `configs/dataset/isles24_base.yaml`
- `configs/dataset/isles24_modalities_baseline.yaml`
- `configs/dataset/isles24_modalities_empirical.yaml`
- `configs/dataset/isles24_modalities_team1.yaml`

## 5.2 Data mode

- `configs/data_mode/online_slices_3d_to_2d.yaml`
- `configs/data_mode/nnunet_slices_2d.yaml`
- `configs/data_mode/full_volumes_3d.yaml`

## 5.3 Data IO

- `configs/data_io/isles24_online_volumes.yaml`
- `configs/data_io/isles24_nnunet_050_baseline.yaml`
- `configs/data_io/isles24_nnunet_052_empirical.yaml`
- `configs/data_io/isles24_nnunet_051_team1.yaml`

Naming convention:

- `isles24_nnunet_{dataset_id}_{idkey}.yaml`

## 5.4 Data runtime

- `configs/data_runtime/cluster_heavy_io.yaml`
- `configs/data_runtime/cluster_balanced_io.yaml`
- `configs/data_runtime/local_debug_io.yaml`

## 5.5 Optional composites

- `configs/data_profile/isles24_2d_online_baseline.yaml`
- `configs/data_profile/isles24_2d_nnunet_baseline.yaml`
- `configs/data_profile/isles24_3d_fullvol_baseline.yaml`

## 6. Loader Routing and Class Intent

Routing source of truth:

- `cfg.data_mode.loader_mode`

Target classes:

- `ISLES24OnlineProc2D`
- `ISLES24NNUNet2D`
- `ISLES24Dataset3D`

## 7. Validation Matrix

| loader_mode | Required keys | Forbidden or ignored keys | Expected output |
|---|---|---|---|
| `online_slices_3d_to_2d` | `dim=2d`, `paths.data_root`, `paths.split_file`, `modalities`, `num_modalities` | `paths.nnunet_*` | image `[C,H,W]`, label `[1,H,W]`, sample id |
| `nnunet_slices_2d` | `dim=2d`, `paths.nnunet_root`, `paths.nnunet_dataset_id`, `paths.nnunet_dataset_name`, `modalities`, `num_modalities` | `paths.split_file` unless explicit extension | image `[C,H,W]`, label `[1,H,W]`, sample id |
| `full_volumes_3d` | `dim=3d`, `paths.data_root`, `paths.split_file`, `modalities`, `num_modalities` | `paths.nnunet_*` | image `[C,H,W,D]`, label `[1,H,W,D]`, sample id |

Global invariants:

- `len(modalities) == num_modalities`
- channel order is stable and consistent
- image and label geometry match per sample

## 8. Core Source Changes Required

- Refactor `src/data/loaders.py` routing to `loader_mode`.
- Add strict config validator.
- Preserve dataloader return structure and training interface contract.
- Preserve sample identity contract used by downstream evaluation.

## 9. Top-Level Experiment Config Composition

Top-level experiment configs must compose explicit groups:

- `/dataset`
- `/data_mode`
- `/data_io`
- `/data_runtime`

This replaces implicit loader behavior encoded via dataset naming.

## 10. Downstream Script Impact

High-impact scripts requiring contract updates:

- `scripts/nnunet/convert_isles24_2d_dataset_to_nnunet.py`
- `scripts/evaluation/compute_segmentation_metrics_for_diffusionmodel_2d_predictions.py`
- `scripts/analysis/threshold_analysis.py`

Medium impact:

- `scripts/evaluation/io_diffusion.py`

Low impact:

- nnUNet CLI runners and nnUNet post-threshold evaluation scripts.

Migration policy:

- No temporary compatibility layer during transition.
- Scripts are aligned after core contract and loader routing are stable.

## 11. Expected Output State

After completion:

- Data loading mode is explicit and validated.
- Config contract is composable and unambiguous.
- nnUNet data source variants are consistently named and profile-aligned.
- Core training and evaluation interfaces remain stable.
- Codebase is prepared for both precomputed 2D and 3D volume pathways.

## 12. Risks and Mitigations

- Risk: contract mismatch between config and loader constructors.  
  Mitigation: centralized validator and hard failures.

- Risk: script breakage due to key path changes.  
  Mitigation: dedicated post-transition script phase.

- Risk: sample identity mismatch in evaluation grouping.  
  Mitigation: enforce normalized identity contract.

- Risk: config sprawl.  
  Mitigation: strict naming and optional `data_profile` composites.

## 13. Assumptions and Dependencies

- nnUNet-converted datasets are valid and usable.
- Hydra remains configuration orchestration mechanism.
- ISLES24 remains primary dataset family in this refactor scope.
- Environment configs continue to provide local/cluster base defaults.

## 14. Acceptance Criteria

1. Three loader modes are represented and validated.
2. Loader dispatch is based only on `loader_mode`.
3. Top-level configs compose explicit data groups.
4. nnUNet data_io naming follows `isles24_nnunet_{id}_{idkey}`.
5. Post-transition scripts run on canonical keys.
6. Documentation reflects canonical data contract.

## 15. Implementation Phases

- Phase A: Config groups and schema scaffolding.
- Phase B: Loader routing + validation refactor.
- Phase C: Experiment config migration.
- Phase D: nnUNet loader backend integration.
- Phase E: Post-transition script alignment.
- Phase F: Cleanup and deprecation.

