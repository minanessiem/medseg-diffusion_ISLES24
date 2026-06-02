# PRD: Subgroup-Aware Scientific Evaluation Pipeline

## Overview And Task Description

The current evaluation pipeline provides stable aggregate segmentation metrics for trained diffusion, discriminative, and nnU-Net-style baselines. It already supports slice-level and reconstructed volume-level metric aggregation, threshold sweeps, selected-threshold summaries, ensemble analysis cases, and optional reconstructed NIfTI exports.

The next step is to turn this aggregate evaluator into a scientific analysis layer that can answer clinically relevant questions about model behavior across cohorts and lesion phenotypes. The key requirement is to report the same canonical metrics not only globally, but also across interpretable strata such as imaging center/site, dataset/source, lesion-volume bucket, foreground/empty case status, split, modality protocol, and model-analysis case.

This should extend the existing evaluation package rather than create a disconnected analysis script. The long-term goal is a reusable evaluation foundation for papers, ablations, challenge submissions, and model comparison reports.

## Current State

The supported campaign evaluation path lives in `scripts/evaluation/`.

Current entry points:

- `scripts/evaluation/compute_segmentation_metrics_for_diffusionmodel_2d_predictions.py`
- `scripts/evaluation/compute_segmentation_metrics_for_nnunet_2d_predictions.py`
- SLURM wrappers under `scripts/evaluation/slurm_runners/`

Current outputs:

- `canonical_results.json`
- `metrics_per_threshold.csv`
- `volume_metrics_per_threshold.csv`
- `evaluation_summary.txt`
- optional reconstructed volumes under `reconstructed_volumes/`

Important existing strengths:

- `SliceSample` and `VolumeSample` already include a `metadata` dictionary in `scripts/evaluation/contracts.py`.
- `DualLevelStreamingMetricsEngine` already computes both slice-level and volume-level metrics in streaming mode.
- The custom-model evaluator already requests dataset metadata when the dataset exposes `return_metadata`.
- `append_per_slice_metrics_rows` exists in `scripts/evaluation/reporting.py`, but is not currently wired into the main entry points.
- The nnU-Net IO path already records file-level metadata such as prediction path, GT path, affine, and shape.

Important gaps:

- Metrics are aggregate-first and do not persist per-case/per-volume metric rows.
- Subgroup metrics are not emitted in JSON or CSV artifacts.
- ISLES26 online 2D can expose `siteID`, `metadata_csv`, and nested metadata, but canonical evaluation outputs do not preserve these fields.
- ISLES24 online 2D currently returns only `(image, label, virtual_path)`, while the ISLES24 nnU-Net slice loader has richer metadata.
- Lesion-size subgrouping requires reliable volume-level ground-truth lesion volume in ml, but current 3D metric defaults use unit spacing unless overridden.
- Missing predictions and skipped mismatched pairs are counted, but not represented as a detailed audit table.

## Requirements And Scope

### Functional Requirements

1. The evaluator must continue to produce the existing aggregate artifacts without breaking current commands.
2. The evaluator must emit per-volume and optionally per-slice metric rows with stable identifiers and normalized metadata.
3. The evaluator must support configurable subgroup analyses at volume level and, where scientifically meaningful, at slice level.
4. Subgroups must include lesion-volume bins by default:
   - `empty`: 0 ml, optional but useful for false-positive analysis
   - `<5ml`
   - `>=5ml_and_<20ml`
   - `>=20ml`
5. Subgroups should also support metadata keys such as:
   - imaging center/site (`siteID`, `center`, or normalized `site_id`)
   - dataset/source (`ISLES24`, `ISLES26`, nnU-Net baseline export, external test set)
   - validation/test split or fold
   - modality protocol or modality set
   - model-analysis case for ensemble studies
6. Subgroup definitions must be configuration-driven and reproducible.
7. Reports must include sample counts per group so that small or unstable strata are visible.
8. The evaluator must preserve its streaming-first behavior for tensor-heavy data.
9. The evaluator should provide a detailed skipped-sample ledger for missing predictions, shape mismatches, invalid metadata, and malformed identities.
10. Threshold selection must remain clear:
    - one global selected threshold for primary comparison by default
    - optional per-group threshold reporting as secondary analysis, not the default headline result

### Non-Functional Requirements

1. The implementation must be modular and testable, with subgroup logic separated from inference, IO, and metric implementations.
2. Core metric computation must remain framework-independent and should not depend on plotting, logging, or paper-specific output code.
3. Output schemas must be stable enough to serve as paper-analysis inputs.
4. Existing commands should remain valid; new features should be opt-in initially.
5. The design should work for both custom model predictions and nnU-Net post-threshold predictions.
6. Artifacts should be easy to compare across model runs without relying on notebooks.

## Components Of Interest

### Existing Evaluation Contracts

- `scripts/evaluation/contracts.py`
  - Extend metadata expectations rather than replacing `SliceSample` or `VolumeSample`.
  - Consider new dataclasses for per-sample metric rows and subgroup definitions.

### IO Adapters

- `scripts/evaluation/io_diffusion.py`
  - Already merges loader metadata into `SliceSample.metadata`.
  - Natural place to normalize per-slice identity, source, split, and dataset tags.
- `scripts/evaluation/io_nnunet.py`
  - Natural place to attach `volume_id`, file paths, affine/spacing, and optional external metadata mapping.

### Metric Engine

- `scripts/evaluation/metrics_engine.py`
  - Current aggregate engine should remain intact.
  - Add either a wrapper engine or a companion accumulator for group-aware metric states.
  - Avoid duplicating inference; subgroup aggregation should consume the same `SliceSample`/`VolumeSample` stream.

### Volume Assembly

- `scripts/evaluation/volume_assembler.py`
  - Currently preserves only `first_slice_metadata` at volume level.
  - Should promote stable per-volume metadata fields such as `case_id`, `site_id`, `split`, `spacing`, and source identity.

### Reporting

- `scripts/evaluation/reporting.py`
  - Add writers for per-volume rows and subgroup CSVs.
  - Extend `canonical_results.json` with a `subgroup_analysis` block.
  - Keep existing aggregate keys stable.

### Dataset Loaders

- `src/data/loader_stack/isles26_loader.py`
  - Already supports metadata passthrough for online 2D.
- `src/data/loader_stack/isles24_loader.py`
  - ISLES24 online 2D should gain optional `return_metadata` support.
  - ISLES24 nnU-Net slice loader already has a metadata path and can inform the online loader shape.

### Existing Tests

- `tests/test_evaluation_reporting.py`
- `tests/test_evaluation_dual_level_engine.py`
- `tests/test_evaluation_io_diffusion.py`
- `tests/test_evaluation_io_nnunet.py`
- `tests/test_evaluation_volume_assembler.py`

These are the right test families to extend for subgroup metadata, per-volume rows, and report schema stability.

## Necessary Changes

### 1. Normalize Evaluation Metadata

Add a small module, for example `scripts/evaluation/metadata.py`, responsible for converting heterogeneous loader and IO metadata into a stable evaluation schema.

Recommended normalized fields:

- `case_id`
- `volume_id`
- `slice_id`
- `slice_index`
- `dataset_id`
- `dataset_name`
- `source_dataset`
- `site_id`
- `fold`
- `split`
- `modality_set`
- `source_path`
- `source_spacing_xyz`
- `source_affine`
- `analysis_case_key`

The module should also support loading an optional external metadata CSV keyed by `case_id` or `volume_id`. This is important for center labels, scanner/protocol variables, demographic variables, and paper-specific cohort annotations that may not belong in the core datalist.

### 2. Add Volume-Level Per-Case Metric Rows

Add a per-volume metric row path that records, at minimum:

- run/model identifiers
- `analysis_case_key`
- `case_id`
- `volume_id`
- threshold
- selected-threshold marker
- GT lesion volume in `mm3` and `ml`
- predicted lesion volume in `mm3` and `ml`
- lesion-volume bin
- all 3D metrics
- normalized metadata fields

This is the most important new artifact for scientific work because most clinically meaningful subgrouping should happen per patient/volume, not per slice.

Proposed output:

- `volume_metrics_per_case.csv`

### 3. Add Subgroup Aggregation

Add a subgroup accumulator that consumes per-volume rows and computes summary stats by:

- threshold
- metric
- subgroup dimension
- subgroup value
- analysis case

Proposed default dimensions:

- `lesion_volume_bin`
- `site_id`
- `source_dataset`
- `split`
- `foreground_status`

Proposed output:

- `volume_metrics_by_group.csv`
- `subgroup_summary.json`

The canonical JSON can include a compact `subgroup_analysis` block, but large subgroup tables should live in CSV to keep the JSON readable.

### 4. Add Optional Per-Slice Rows

Per-slice outputs are useful for debugging and threshold behavior, but less useful as primary scientific evidence because slices are not independent patient samples.

Proposed output:

- `slice_metrics_per_sample.csv`

This should be opt-in or controlled by a flag such as `--export-per-slice-metrics` to avoid unnecessarily large artifacts.

### 5. Fix Physical-Unit Volume Metrics

Current volume metrics default to unit spacing. For lesion-size bins and mm-based metrics to be scientifically meaningful, the volume evaluator should pass real voxel spacing when available.

Recommended behavior:

- Prefer spacing from normalized metadata (`source_spacing_xyz`).
- Fall back to affine-derived spacing when available.
- If no spacing is available, emit an explicit warning and mark `spacing_source=unit_fallback`.
- Store `spacing_xyz`, `spacing_source`, and `voxel_volume_mm3` in per-volume rows.

For resized 2D model inputs, the design must be explicit about whether metrics are measured in resized model space or source physical space. The safest near-term contract is to report resized-space metrics by default and attach a prominent `physical_units_valid` flag unless predictions/GT are reconstructed in source geometry.

### 6. Add Config And CLI Surface

Recommended flags for both main entry points:

- `--metadata-csv <path>`
- `--metadata-case-key case_id`
- `--group-by lesion_volume_bin,site_id,source_dataset`
- `--lesion-volume-bins 0,5,20`
- `--export-per-volume-metrics`
- `--export-per-slice-metrics`
- `--min-group-size 5`
- `--subgroup-threshold-mode global_selected|all_thresholds`

For Hydra-based workflows, mirror the same options in an evaluation config group later, for example `configs/evaluation/scientific_subgroups.yaml`.

### 7. Add Scientific Report Outputs

Add a concise human-readable report that includes:

- global selected-threshold metrics
- volume-level lesion-size subgroup table
- center/site subgroup table
- group counts and warnings for low-count strata
- skipped-sample summary
- threshold protocol summary
- physical-unit validity notes

Proposed output:

- `scientific_evaluation_summary.md`

This should be generated from CSV/JSON artifacts, not from hidden in-memory state, so it can be regenerated or extended later.

## Expected Output State

After implementation, a standard custom-model evaluation run should produce the current outputs plus opt-in subgroup artifacts:

- `canonical_results.json`
- `metrics_per_threshold.csv`
- `volume_metrics_per_threshold.csv`
- `volume_metrics_per_case.csv`
- `volume_metrics_by_group.csv`
- `subgroup_summary.json`
- `scientific_evaluation_summary.md`
- optional `slice_metrics_per_sample.csv`
- optional `skipped_samples.csv`
- optional reconstructed volumes

The resulting pipeline should answer questions such as:

- Does model Dice degrade for lesions under 5 ml?
- Does false-positive burden differ across empty or low-lesion-volume cases?
- Is performance stable across imaging centers?
- Are threshold choices robust across lesion-size strata?
- Do ensemble methods improve small-lesion recall or mainly improve larger lesions?
- Are nnU-Net and diffusion baselines failing on different subgroups?

## Recommended Analysis Dimensions

### Primary Paper-Ready Dimensions

- Lesion volume: `empty`, `<5ml`, `>=5ml_and_<20ml`, `>=20ml`
- Imaging center/site: normalized `site_id`
- Dataset/source cohort: `ISLES24`, `ISLES26`, external validation set, nnU-Net export source
- Foreground status: empty vs foreground volumes
- Model-analysis case: ensemble size and method

### Secondary Dimensions

- Number of connected lesions
- Small-lesion count
- Slice count per volume
- Modality set
- Acquisition/protocol fields from external metadata
- Fold/split
- Scanner/vendor fields if available and approved for use

### Statistics To Consider

Initial implementation should report mean, std, and count to match existing running stats.

A later scientific-reporting phase should add:

- median
- interquartile range
- bootstrap confidence intervals
- paired model deltas when comparing two runs on the same case IDs
- nonparametric test outputs only when the comparison design is explicit

## Risks And Mitigations

### Risk: Misleading Per-Slice Scientific Conclusions

Per-slice metrics can overweight large lesions or long volumes.

Mitigation:

- Treat volume-level/per-case rows as the primary scientific artifact.
- Label per-slice outputs as debugging or threshold-analysis artifacts.

### Risk: Incorrect Physical Units

Lesion-volume bins require reliable voxel spacing and geometry. Current 3D defaults use unit spacing.

Mitigation:

- Add explicit spacing provenance fields.
- Warn or fail when lesion-volume subgrouping is requested without valid spacing, depending on a strictness flag.
- Prefer source-space reconstruction for final paper metrics when available.

### Risk: Metadata Inconsistency Across Datasets

ISLES24 and ISLES26 loaders expose different metadata fields.

Mitigation:

- Normalize metadata through one module.
- Support external metadata CSV joins.
- Use `unknown` group values rather than dropping cases silently.

### Risk: Group Counts Too Small For Reliable Claims

Center and lesion-size strata may contain few cases.

Mitigation:

- Always report group counts.
- Add `min_group_size` warnings.
- Keep low-count groups visible but flagged.

### Risk: Stack Divergence

There are overlapping evaluation paths under `scripts/evaluation/`, `scripts/analysis/`, and `scripts/nnunet/`.

Mitigation:

- Implement subgroup-aware artifacts first in `scripts/evaluation/`, which is documented as the supported campaign path.
- Reuse outputs in later analysis/figure scripts rather than duplicating metric computation.

## Assumptions And Dependencies

Assumptions:

- Scientific headline metrics should be volume-level whenever possible.
- The existing `scripts/evaluation/` package remains the canonical evaluation path.
- The first implementation should be backward compatible with existing commands.
- External cohort metadata may exist outside the datalist and should be joinable at evaluation time.

Dependencies:

- No new required Python dependency is needed for the first implementation.
- Optional bootstrap confidence intervals can be implemented with NumPy/Pandas if already available, or deferred until the artifact schema is stable.
- Any future statistical testing should be added only after model-comparison requirements are explicit.

## Proposed Implementation Phases

1. Metadata normalization and optional metadata CSV join.
2. ISLES24 online 2D `return_metadata` support.
3. Per-volume metric row export at selected and/or all thresholds.
4. Lesion-volume calculation, binning, and spacing provenance.
5. Group-aware aggregation from per-volume rows.
6. Report writers for subgroup CSV/JSON/Markdown outputs.
7. Optional per-slice rows for debugging and threshold analysis.
8. Tests for metadata normalization, per-volume rows, subgroup aggregation, and backward-compatible entry points.

Implementation should not proceed until this PRD is reviewed and approved.
