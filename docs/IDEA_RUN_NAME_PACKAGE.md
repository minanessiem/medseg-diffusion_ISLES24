# Idea: Configurable Run Name Package

**Status:** Idea / future work  
**Priority:** Deferred until higher-priority experiment and modeling work is complete  
**Motivation:** Turn run naming from a growing utility function into a modular,
configurable experiment metadata subsystem.

## Overview

`src/utils/run_name.py` has grown beyond a simple helper. It now encodes naming
logic for model architecture, dataset preprocessing, augmentation, optimizer,
scheduler, loss, AMP, validation ensemble, diffusion settings, and timestamps.

As the project adds more experiment axes, the current single-file approach makes
new naming policies harder to reason about and harder to test. A dedicated run
name package would make each token source modular while allowing the final run
name layout to be configured per workflow.

## Goals

- Split run-name token generation by config group.
- Centralize dict/OmegaConf access through shared helpers.
- Make token behavior independently testable.
- Make final run-name composition configurable through Hydra.
- Preserve the current naming policy as the default for backward compatibility.
- Make active experimental variables visible in `squeue`, output folders, and
  experiment summaries.

## Proposed Package Layout

```text
src/utils/run_names/
  __init__.py
  builder.py
  config_access.py
  registry.py
  tokens/
    __init__.py
    augmentation.py
    dataset.py
    diffusion.py
    loss.py
    model.py
    optimizer.py
    scheduler.py
    training.py
    validation.py
```

Potential responsibilities:

- `builder.py`: assemble tokens into the final run name.
- `config_access.py`: shared helpers for plain dicts and OmegaConf configs.
- `registry.py`: map configured token names to token builder functions.
- `tokens/*.py`: own compact naming logic for one config group or concern.

## Config-Driven Naming

Add a config group such as:

```text
configs/run_name/
  default.yaml
  discriminative_3d.yaml
  diffusion_2d.yaml
  nnunet_baseline.yaml
```

Example default policy:

```yaml
run_name:
  separator: "_"
  include_empty: false
  unknown_token_policy: error
  parts:
    - model
    - batch
    - context
    - amp
    - optimizer
    - clip
    - scheduler
    - steps
    - loss
    - dataset_preprocessing
    - augmentation
    - diffusion
    - validation_ensemble
    - timestamp
```

Example experiment-focused policy:

```yaml
run_name:
  separator: "_"
  include_empty: false
  parts:
    - model
    - dataset_preprocessing
    - augmentation
    - steps
    - loss
    - timestamp
```

This would allow different workflows to keep names concise while still exposing
the variables under observation.

## Compatibility Plan

Preserve the current public import path initially:

```python
from src.utils.run_names.builder import generate_run_name
```

Then keep `src/utils/run_name.py` as a compatibility shim:

```python
from src.utils.run_names.builder import generate_run_name
```

The first implementation should reproduce the current default output format so
existing scripts, logs, and output assumptions do not break.

## Token Examples

Dataset preprocessing:

```text
T1_RAW        -> t1RAW
T1_ZSCORE     -> t1ZSC
T1_PCTNORM    -> t1PCT
T1_PCT_ZSCORE -> t1PZSC
```

Augmentation:

```text
none                         -> augNONE
light_3d                     -> augLIGHT3D
aggressive_3d                -> augAGG3D
spatial_only_3d              -> augSPAT3D
pctnorm_light_intensity_3d   -> augPCTLITE3D
zscore_safe_3d               -> augZSAFE3D
raw_spatial_plus_blur_3d     -> augRAWBLUR3D
raw_scaled_intensity_3d      -> augRAWSCALE3D
```

## Relationship To Experiment Management

This idea complements `docs/IDEA_EXPERIMENT_MANAGEMENT_COMPONENT.md`.

The experiment management component would define and submit coherent groups of
runs. The run-name package would ensure each run exposes the right variables in
its folder name, scheduler job name, and result summaries.

Together they support a workflow where experiments are:

1. declared in config,
2. submitted as tracked run groups,
3. named according to the active variables,
4. summarized automatically after completion.

## Open Questions

- Should unknown token names fail fast or be skipped?
- Should token names be registered statically, dynamically, or through config?
- Should run-name policies be selected manually or inferred from experiment
  family?
- Should experiment ID or tier be included in run names, output roots, or only
  manifests?
- Should long generated names have automatic shortening?
- How should multi-modality datasets be encoded without making names too long?

## Deferred Related Work

- Add a shared dict/OmegaConf config accessor and remove repeated compatibility
  branches from `src/utils/run_name.py`.
- Add tests for each token module and for policy-level run-name assembly.
- Integrate run-name policies with the future experiment runner.
