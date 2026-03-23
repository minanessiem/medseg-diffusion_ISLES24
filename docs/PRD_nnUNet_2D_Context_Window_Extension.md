# PRD: nnUNet 2D Context-Window Extension

Version: 1.0  
Date: 2026-03-23  
Status: Implemented on `experiment/prd-nnunet-2d-context-window`

## 1. Overview

This change extends `nnunet_slices_2d` with configurable neighboring-slice context while preserving a strict 2D training contract.

- Loader mode remains `nnunet_slices_2d`
- Dim remains `2d`
- nnUNet dataset format remains unchanged
- Model input stays `[B, C, H, W]` with flattened channels

## 2. New Data Contract Fields

Under `configs/data_mode/nnunet_slices_2d.yaml`:

- `per_side_context_slices`: integer `>= 0`
  - `0` means legacy behavior (center slice only)
  - `k > 0` loads `k` neighbors before and after center
- `channel_layout`: `slice_major` or `modality_major`
  - controls flattening order of `[modality, context_slice]` into channels

Derived terms:

- `num_effective_slices = 2 * per_side_context_slices + 1`
- `effective_input_channels = dataset.num_modalities * num_effective_slices`

## 3. Runtime Behavior

### 3.1 Boundary policy

If neighbor slices are out of bounds for a volume, their channels are zero-filled.

### 3.2 Label policy

Segmentation target remains center slice only (`[1, H, W]`).

### 3.3 Validation

`validate_dataset_contract` enforces:

- context fields are required/validated for `nnunet_slices_2d`
- context fields are rejected for non-nnUNet loader modes in this phase

## 4. Model Channel Contract

Training bootstrap now validates `cfg.model.image_channels` against the active data contract:

- for `nnunet_slices_2d`: expects `effective_input_channels`
- for other modes: expects `dataset.num_modalities`

The training bootstrap also auto-syncs `model.image_channels` to this expected value before
model construction. This prevents silent mismatches at startup while keeping config overrides optional.

## 5. Logging Behavior

For context runs (`per_side_context_slices > 0`) in `nnunet_slices_2d`:

- the left input block is rendered as one compact mini-grid panel
- mini-grid shape corresponds to `[num_modalities x num_effective_slices]`
- the right-side panels (target/noise/prediction/snapshots) remain single-slice panels

Applied to:

- forward-step image logging
- sampling snapshots logging
- ensembled segmentation logging

## 6. Run Name Encoding

Run names now include context metadata for nnUNet 2D runs:

- `ctxps{k}`: per-side context slices
- `slmaj` or `mdmaj`: channel flattening layout

Example token: `ctxps2_mdmaj`

## 7. Usage Examples

Legacy-equivalent:

```yaml
data_mode:
  loader_mode: nnunet_slices_2d
  dim: 2d
  per_side_context_slices: 0
  channel_layout: slice_major
```

Context run (`k=1`, 3 effective slices):

```yaml
data_mode:
  loader_mode: nnunet_slices_2d
  dim: 2d
  per_side_context_slices: 1
  channel_layout: slice_major
```

Context run (`k=2`, 5 effective slices, modality-major):

```yaml
data_mode:
  loader_mode: nnunet_slices_2d
  dim: 2d
  per_side_context_slices: 2
  channel_layout: modality_major
```

CLI override example:

```bash
python3 -m start_training \
  --config-name local_nnunet2d_baseline \
  data_mode.per_side_context_slices=1 \
  data_mode.channel_layout=slice_major
```

## 8. Tests Added

- `tests/test_run_name_context_tokens.py`
- `tests/test_nnunet_context_loader.py`

Validated via:

- `python3 -m unittest tests.test_run_name_context_tokens -v`
- `python3 -m unittest tests.test_nnunet_context_loader -v`
