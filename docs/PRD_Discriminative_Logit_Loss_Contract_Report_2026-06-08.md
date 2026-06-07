# PRD Report: Discriminative Logit-Native Loss Contract

**Date:** 2026-06-08  
**Branch:** `feature/discriminative-logit-loss-contract`  
**Status:** Implementation drafted; compute-environment validation pending

## Overview

This report documents the final PRD scope and implementation state for the
discriminative logit-native loss contract. The change is focused on ISLES26
discriminative segmentation experiments with DynUNet and SwinUNETR.

The core design decision is:

- DynUNet and SwinUNETR adapters return raw segmentation logits.
- The discriminative loss engine routes each configured loss term to either
  logits or sigmoid probabilities via an explicit `input_domain` field.
- Public inference paths used by validation, metrics, and image logging return
  probabilities.
- Generative MedSegDiff/OpenAI diffusion training and `auxiliary_losses` remain
  untouched.

This makes logits-native losses possible for future work while preserving the
current probability-domain behavior of existing discriminative Dice/BCE configs.

## Final PRD Scope

### In Scope

- Migrate all `configs/loss/discriminative_*.yaml` files to explicit active
  `terms`.
- Remove legacy `discriminative.dice` / `discriminative.bce` schema support.
- Remove term-level `enabled`; term presence means active.
- Remove redundant term `name`; the configured `loss` class key identifies the
  term.
- Use class-style loss identifiers such as `DiceLoss` and `BCELoss`.
- Require every term to declare:
  - `loss`
  - `input_domain`
  - `weight`
  - `params`
  - `supervision`
- Make DynUNet and SwinUNETR adapters logit-native.
- Keep `DiscriminativeAdapter.sample()` and `sample_with_snapshots()` returning
  probabilities.
- Add tests for schema validation and logits/probability domain routing.

### Out Of Scope

- No changes to MedSegDiff/OpenAI diffusion.
- No changes to `auxiliary_losses`.
- No softmax/multi-class activation policy yet.
- No new loss functions in this PRD.
- No threshold-sweep work.

## Files Changed

### Loss Configs

- `configs/loss/discriminative_dicebce.yaml`
- `configs/loss/discriminative_dice_only.yaml`
- `configs/loss/discriminative_dicebce_deepsupervision.yaml`
- `configs/loss/discriminative_dice_only_deepsupervision.yaml`

All four configs now use explicit `discriminative.terms`.

Current migrated behavior:

- Existing `DiceLoss` terms use `input_domain: probabilities` and
  `apply_sigmoid: false`.
- Existing `BCELoss` terms use `input_domain: probabilities` and
  `apply_sigmoid: false`.
- Dice-only configs contain only a `DiceLoss` term.
- Dice+BCE configs contain `DiceLoss` and `BCELoss` terms.

### Model Adapters

- `src/models/DynUNet/model_adapter.py`
- `src/models/SwinUNetR/model_adapter.py`

Both adapters now return raw logits directly from the wrapped MONAI model.
Docstrings were updated to state that probability conversion is downstream.

### Discriminative Loss Engine

- `src/losses/discriminative_deep_supervision.py`

Implemented changes:

- Removed hidden default loss factory behavior.
- Removed legacy `dice`/`bce` parsing.
- Added explicit class-style loss registry:
  - `DiceLoss`
  - `BCELoss`
- Added required loss parameter validation.
- Added schema validation for unsupported fields such as legacy `enabled` or
  `name`.
- Added `input_domain` routing:
  - `logits`: raw model-output heads
  - `probabilities`: sigmoid-transformed heads
- Returned `final_prediction` and `head_predictions` in probability domain for
  compatibility with logging and downstream display.
- Added a TODO documenting that sigmoid conversion assumes current binary
  single-channel segmentation.

### Discriminative Adapter

- `src/diffusion/discriminative_adapter.py`

Implemented changes:

- Updated initialization logging to use `term.loss_key`.
- Added centralized logits-to-probability conversion for public inference paths.
- Updated `sample()` to return probabilities from logits.
- Updated `sample_with_snapshots()` to yield probabilities from logits.
- Clarified in comments that losses may consume logits/probabilities, while
  metrics/logging consume probabilities.

### Tests

- `tests/test_discriminative_output_domains.py`
- `tests/test_swinunetr_adapter.py`

Implemented test changes:

- Added focused tests for explicit discriminative schema validation.
- Added tests for rejecting legacy `dice`/`bce`, `enabled`, and `name` fields.
- Added tests for logits/probability routing in
  `compute_discriminative_deep_supervision_loss()`.
- Added tests that `DiscriminativeAdapter.sample()` and
  `sample_with_snapshots()` return probabilities.
- Updated SwinUNETR adapter tests to expect finite logits instead of `[0, 1]`
  probabilities.

## Current Validation Status

Completed locally:

- `ReadLints` reports no linter errors on edited source/config/test files.
- Static searches found no stale references to:
  - `term.name`
  - `term.enabled`
  - legacy builders
  - lowercase `loss: dice` / `loss: bce`
  - legacy term `name: dice` / `name: bce`

Not completed locally:

- `pytest` execution was intentionally not completed in this environment. The
  full test suite and focused tests should be run in a compute-capable
  environment.

## Required Validation Before PRD Completion

Run the focused tests first:

```bash
python3 -m pytest tests/test_discriminative_output_domains.py
python3 -m pytest tests/test_swinunetr_adapter.py
python3 -m pytest tests/test_loss_metric_alignment.py
```

Then run broader regression tests:

```bash
python3 -m pytest tests/
```

Recommended config composition checks:

```bash
python3 -m main --config-name cluster_isles26_3d_randompatch_dynunet --help
python3 -m main --config-name cluster_isles26_3d_randompatch_swinunetr --help
python3 -m main --config-name local_isles26_3d_randompatch_dynunet --help
python3 -m main --config-name local_isles26_3d_randompatch_swinunetr --help
```

If `main --help` does not compose Hydra configs in this project, use the
project's standard config-composition or smoke-test command instead.

Recommended runtime smoke tests:

- One short DynUNet discriminative run with `discriminative_dicebce_deepsupervision`.
- One short SwinUNETR discriminative run with `discriminative_dicebce`.
- At least one validation pass to confirm:
  - `sample()` returns probabilities.
  - `dice_3d`, `surface_dice_monai_3d`, and `hd95_3d` still compute normally.
  - TensorBoard/image logging receives probability masks.

Suggested runtime overrides:

```bash
python3 -m scripts.slurm.single_job_runner \
  --config-name local_isles26_3d_randompatch_dynunet \
  --overrides training.max_steps=2 validation.validation_interval=1
```

Adjust the exact command to the active local runner conventions.

## Acceptance Criteria

The PRD can be considered complete when:

- All focused tests pass.
- Full relevant test suite passes or failures are confirmed unrelated.
- ISLES26 DynUNet and SwinUNETR configs compose successfully.
- A short discriminative training smoke test completes without schema or tensor
  domain errors.
- Validation metrics still receive probability predictions.
- No unintended changes are made to MedSegDiff/OpenAI diffusion behavior.

## Follow-Up Work

After this PRD is validated, the codebase is ready for new discriminative loss
configs and implementations, such as:

- logits-domain `BCELoss` with `apply_sigmoid: true`;
- MONAI `DiceFocalLoss`;
- MONAI `TverskyLoss`;
- MONAI `HausdorffDTLoss`;
- custom boundary-band BCE/logit losses.

Future multi-class segmentation should add an explicit activation policy
instead of assuming sigmoid probability conversion.
