# Idea: Experiment Management Component

**Status:** Idea / future work  
**Priority:** Deferred until higher-priority modeling work is complete  
**Motivation:** Keep large SLURM experiment grids organized, cancellable, and easy to summarize.

## Overview

As the project moves toward larger experiment grids, individual run submission is
becoming too low-level as the main operational unit. The useful unit is an
experiment: a coherent set of runs that share a hypothesis, output root, base
configuration, comments, and expected decision criteria.

This component would build on the existing `scripts/slurm/single_job_runner.py`
workflow and add a higher-level `experiment_runner` that submits, tracks, and
summarizes groups of runs.

## Goals

- Define experiments as config-driven groups of related runs.
- Keep the SLURM queue productively filled without losing experiment context.
- Record submitted job IDs so an entire experiment can be cancelled easily.
- Store human-readable hypotheses, comments, and notes next to run outputs.
- Provide scripts to aggregate best metrics from completed runs.
- Export summaries in console, CSV/JSON, Markdown, and eventually LaTeX formats.

## Proposed Config Shape

Add an experiment config group such as:

```text
configs/experiments/
  isles26_dynunet_augscale_100k.yaml
  isles26_nnunet_preprocess_baseline.yaml
```

Example:

```yaml
experiment:
  id: isles26_dynunet_augscale_100k_06062026
  tier: tier1
  output_root: /mnt/outputs/dynunet_isles26_augscale_100K_06062026
  base_config: cluster_isles26_3d_randompatch_dynunet
  summary: >
    Tests whether scale-aware augmentation changes the RAW vs normalized
    preprocessing conclusion for DynUNet on ISLES26.

  runs:
    - name: raw_spatial_only
      overrides:
        augmentation: spatial_only_3d
    - name: pctnorm_spatial_only
      overrides:
        data_profile: isles26_3d_randompatch_t1pctnorm
        augmentation: spatial_only_3d
```

The `tier` field is intended for queue management. For example:

- `tier1`: direct hypothesis tests or high-priority follow-ups.
- `tier2`: likely useful ablations.
- `tier3`: speculative runs that may be cancelled if better follow-ups appear.

## Proposed Experiment Directory

Each experiment should produce a durable directory containing both machine- and
human-readable metadata:

```text
/mnt/outputs/<experiment_id>/
  experiment.yaml
  summary.md
  submitted_commands.sh
  jobs.csv
  cancel_all.sh
  results/
    metrics_summary.txt
    metrics_summary.csv
    metrics_summary.json
    metrics_summary.tex
```

`jobs.csv` should include at minimum:

- run alias,
- generated run name,
- SLURM job ID,
- base config,
- overrides,
- output directory,
- submission timestamp,
- current or last-known status.

## Proposed Scripts

Potential entrypoints:

```bash
python3 -m scripts.slurm.experiment_runner --experiment-name isles26_dynunet_augscale_100k
python3 -m scripts.slurm.cancel_experiment --experiment-dir /mnt/outputs/<experiment_id>
python3 -m scripts.experiments.summarize --experiment-dir /mnt/outputs/<experiment_id> --format txt
python3 -m scripts.experiments.summarize --experiment-dir /mnt/outputs/<experiment_id> --format latex --show-comments
```

The runner should reuse the same config composition and SLURM submission logic
as `single_job_runner.py` rather than duplicating command construction.

## Result Aggregation

The summarizer should scan each run directory for existing artifacts such as:

- `.hydra/config.yaml`,
- validation metrics,
- best-checkpoint metadata,
- metric CSV/JSON files,
- TensorBoard scalar exports if no simpler source is available.

Initial summaries should report:

- best validation metric,
- step of best validation metric,
- final validation metric,
- output directory,
- run alias,
- preprocessing token,
- augmentation token,
- model architecture,
- git commit hash
- notes/comments.

## Design Principles

- Keep experiment definitions declarative and version-controlled.
- Preserve exact submitted commands for reproducibility.
- Make cancellation easy for speculative runs.
- Keep enough queue headroom for urgent high-priority follow-ups.
- Avoid making the experiment runner responsible for training logic.
- Treat result aggregation as read-only analysis over produced artifacts.

## Open Questions

- Should experiment metadata be written under the shared output root, under
  `docs/experiments/`, or both?
- Should run aliases be injected into Hydra overrides and run names?
- Which metric artifact should be considered canonical for discriminative 3D
  runs?
- How should partially completed or cancelled runs appear in summaries?
- Should queue tiering be passive metadata only, or should scripts act on it?
