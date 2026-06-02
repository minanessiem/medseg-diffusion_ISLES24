# Handover: Training Reliability, Restart Semantics, and Resume Strategy (2026-06-02)

## Why this document exists

This memo captures the high-signal outcomes from the recent thread:

- DynUNet + deep supervision integration status and audit notes.
- The `ancdata` DataLoader failure investigation and what we learned.
- Config and run-command decisions made during troubleshooting.
- Checkpoint boundary and restart-state design discussion.
- Research references (links) and practical search terms for follow-up work.

The intent is to preserve decision context and avoid re-litigating the same investigation.

---

## 1) Project-direction synthesis (important non-code decision)

Core alignment reached in the thread:

- The codebase is a **research toolkit**.
- Engineering is the **vehicle**, not the end goal.
- Priority is **scientifically publishable experiments** with trustworthy results.
- Platform improvements are valuable when they improve:
  - experiment validity,
  - experiment throughput,
  - or failure/restart reliability.

Practical operating model discussed:

- Default to "experiment mode" (minimal disruptive changes).
- Timebox infrastructure work.
- Prefer risk-reduction changes over open-ended system sophistication.

---

## 2) DynUNet + deep-supervision work: status and notable outcomes

### Major integration scope (already implemented prior to this memo)

- Added DynUNet support and registration in model factory.
- Added generic discriminative deep-supervision engine in losses.
- Wired discriminative adapter to use generic deep-supervision loss compute.
- Added DS-aware run-name tokens and DynUNet model-string support.
- Added DS-enabled config profiles and DS loss config variants.

### Important bug discovered in audit

**DiscriminativeAdapter wrapper-awareness bug**:

- In wrapped models (`DataParallel`/`DDP`), custom attrs like `spatial_dims` may not be exposed on wrapper.
- This could mis-resolve inference parser mode and incorrectly interpret 3D output rank.

Action taken:

- Updated `DiscriminativeAdapter` to resolve metadata from unwrapped model and config fallback.
- Hardened parser fallback behavior for ambiguous rank cases.

---

## 3) Config decisions made during the thread

### Cluster run profiles discussed

- `configs/cluster_isles26_3d_randompatch_dynunet.yaml`
- `configs/cluster_isles26_3d_randompatch_swinunetr.yaml`

### Decisions made

1. Validation subset behavior:
   - Confirmed default inherited from ISLES26 dataset base was `val_fast`.
   - Updated both cluster configs to:
     - `dataset.active_subsets.val: val_full`

2. Worker count adjustments:
   - Initially reduced `num_valid_workers`.
   - Then explicitly set both:
     - `data_runtime.num_train_workers: 8`
     - `data_runtime.num_valid_workers: 8`

3. Validation metrics profile:
   - Both cluster configs continue using:
     - `validation: sliding_window_3d_metrics_subset`

---

## 4) Workload failure investigation (`RuntimeError: received 0 items of ancdata`)

### Observed symptom pattern

- Validation loop often progressed far or completed.
- Crash then occurred around:
  - post-validation image logging path,
  - while fetching from a DataLoader iterator.

### Key diagnostic conclusion

The critical failure was not always from validation workers.

In the logged stack, crash occurred at:

- `step_based_train()` image-logging section,
- during `next(dl_iter)` from `train_dataloader`.

So the failure was in multiprocessing DataLoader IPC path during train-loader fetch, not necessarily validation-loader fetch.

### Why changing `num_valid_workers` alone did not fully resolve

- Train loader remained high worker count in some runs (observed resolved config showed `num_train_workers=32` in at least one run).
- Non-persistent multiprocess workers + prefetch + large 3D data can still hit FD/IPC/shared-memory pressure.
- `ancdata` errors are commonly correlated with resource-handle transfer failures in multiprocessing.

### Immediate stabilization knobs discussed

- Lower worker pressure on the path that crashes.
- Especially tune:
  - `data_runtime.num_train_workers`
  - `data_runtime.train_prefetch_factor`
  - `logging.enable_image_logging` (temporarily disable to isolate)

---

## 5) Checkpoint timing: current vs. ideal location

### Current interval save behavior

`save_interval_checkpoint(...)` is called near end of each macro-step loop iteration in `step_based_train()`.

At save time, model-side state is generally coherent (optimizer/scaler/EMA/scheduler updated), but:

- validation and logging blocks have already run before interval save in many paths,
- so crashes in those blocks can still lose progress since last save.

### Recommended checkpoint boundary (agreed conceptually)

For robust fail-stop restart:

- Save interval checkpoint **immediately after training-step commit**:
  - after optimizer/scaler step,
  - after gradient zeroing,
  - after global step increment,
  - before validation/image/sampling logging side paths.

Metric-based "best checkpoint" remains tied to validation outputs and should stay in validation path.

---

## 6) Restart semantics: what is currently saved vs what is missing

### Currently saved in interval training state

From `checkpoint_utils.save_interval_checkpoint(...)`:

- `global_step`
- `ema_rates`
- `ema_params`
- `scheduler_state_dict`
- `best_metric_value`
- `best_metric_step`
- `scaler_state_dict` (if AMP scaler used)

Also saves model and optimizer state dict files.

### Not currently saved (important for deterministic replay)

- Python RNG state
- NumPy RNG state
- Torch CPU RNG state
- Torch CUDA RNG states
- Dataloader/sampler cursor state
- Sampler epoch and exact next-batch position
- Iterator micro-state/in-flight prefetch state (not realistically restorable)
- Loop bookkeeping like accumulation and interval accumulators (if needed for exact replay)

### Bottom line

Current resume is good for model/optimizer continuation, but **not full deterministic data-stream replay**.

---

## 7) Recovery model options explored

### Option A: in-process soft recovery after worker death

Pros:

- Potentially avoids full process restart.

Cons:

- High risk of hidden state leakage and silent divergence.
- Hard to guarantee reproducibility.
- Difficult to reason about worker-local/in-flight state.

### Option B: fail-stop + restart from last committed checkpoint (recommended)

Pros:

- Simpler correctness model.
- Industry-standard pattern in distributed training.
- Easier to enforce invariant checks and auditability.

Cons:

- Loses work since latest save boundary.

Thread conclusion leaned toward Option B for research reliability.

---

## 8) "Secure against bad hidden state" safeguards discussed

If implementing restart hardening, add these controls:

1. **Fail closed** on loader-worker IPC failure (do not silently continue in same process).
2. **Strict resume contract** with invariant checks:
   - config hash consistency,
   - expected step/LR/scaler sanity,
   - sampler/cursor restore checks.
3. **Save and restore RNG states**.
4. **Restore data cursor state** (sampler/loader stateful mechanism).
5. Optional **next-batch fingerprint check** after resume:
   - compare expected sample-id signature with resumed signature,
   - hard fail on mismatch.
6. Isolate non-critical logging data paths so they cannot poison main training flow.

---

## 9) Commands and operational snippets used/discussed

### SLURM single-job runner (module invocation)

```bash
python3 -m scripts.slurm.single_job_runner --config-name cluster_isles26_3d_randompatch_dynunet
python3 -m scripts.slurm.single_job_runner --config-name cluster_isles26_3d_randompatch_swinunetr
```

### Example worker override usage

```bash
python3 -m scripts.slurm.single_job_runner --config-name cluster_isles26_3d_randompatch_dynunet --overrides data_runtime.num_valid_workers=2
python3 -m scripts.slurm.single_job_runner --config-name cluster_isles26_3d_randompatch_swinunetr --overrides data_runtime.num_valid_workers=2
```

---

## 10) Research/reference links collected during thread

### A) `ancdata` / FD / DataLoader worker IPC failure references

- https://github.com/pytorch/pytorch/issues/973
- https://github.com/pytorch/pytorch/issues/11201
- https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999
- https://stackoverflow.com/questions/71642653/how-to-resolve-the-error-runtimeerror-received-0-items-of-ancdata
- https://stackoverflow.com/questions/48250053/pytorchs-dataloader-too-many-open-files-error-when-no-files-should-be-open

### B) PyTorch reproducibility and worker seeding references

- https://github.com/pytorch/pytorch/blob/main/torch/utils/data/dataloader.py
- https://github.com/pytorch/pytorch/blob/ec673ecd/torch/utils/data/_utils/worker.py
- https://discuss.pytorch.org/t/dataloaders-multiprocess-with-torch-manual-seed/123044
- https://stackoverflow.com/questions/67196075/pytorch-dataloader-uses-identical-random-transformation-across-each-epoch

### C) Fault-tolerant restart patterns (PyTorch/TorchElastic)

- https://docs.pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html
- https://docs.pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.md
- https://docs.pytorch.org/docs/stable/elastic/train_script.md
- https://docs.pytorch.org/docs/stable/elastic/run.md

### D) Checkpoint performance and async checkpointing

- https://pytorch.org/blog/reducing-checkpointing-times/
- https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.md

### E) Stateful dataloaders and mid-epoch resume

- https://meta-pytorch.org/data/main/torchdata.stateful_dataloader.html
- https://github.com/pytorch/data/blob/main/torchdata/stateful_dataloader/README.md
- https://github.com/Lightning-AI/lightning/issues/17105
- https://github.com/Lightning-AI/pytorch-lightning/pull/19361

### F) Framework-level resume behavior references

- Hugging Face Trainer docs:
  - https://huggingface.co/docs/transformers/en/main_classes/trainer
  - https://huggingface.co/docs/transformers/main/en/main%5Fclasses/trainer
- HF forum thread:
  - https://discuss.huggingface.co/t/resume-from-checkpoint-skipping-batches-why-does-the-processing-function-need-to-be-run-for-skipped-batches/31291
- Composer autoresume notes:
  - https://docs.mosaicml.com/projects/composer/en/latest/notes/resumption.html
  - https://docs.mosaicml.com/projects/composer/en/stable/notes/resumption.html
  - https://docs.mosaicml.com/projects/composer/en/v0.32.0/examples/checkpoint_autoresume.html

### G) Supplemental engineering writeups (non-official)

- https://deepiix.com/blog/fault-tolerant-training-systems
- https://mbrenndoerfer.com/writing/checkpointing-recovery-async-fault-tolerance-training
- https://buildai.substack.com/p/checkpointing-and-resumption-stateful
- https://training-api.cerebras.ai/en/rel-2.2.0/wsc/tutorials/dataloader-checkpointing.html

Note: Prefer official framework docs and code references for implementation-critical decisions.

---

## 11) Search-term library (copy/paste ready)

- `pytorch received 0 items of ancdata file descriptor sharing strategy`
- `torchrun fault tolerant training max-restarts checkpoint resume`
- `stateful dataloader mid-epoch checkpointing`
- `sampler state_dict load_state_dict reproducible resume`
- `deterministic replay training after crash rng dataloader cursor`
- `persistent_workers prefetch_factor dataloader stability`
- `resume_from_checkpoint data skip reproducibility`
- `distributed checkpoint async save training throughput`
- `data pipeline checkpointing worker state aggregation`

---

## 12) Recommended near-term plan (publishability-first)

1. Keep experiment configs stable for current paper runset.
2. Use fail-stop restart from interval checkpoints (no soft in-process continuation on worker IPC failure).
3. Move interval save to post-step commit boundary before validation/logging side paths.
4. Add minimal resume hardening in phases:
   - Phase 1: save/restore RNG states + key loop counters.
   - Phase 2: sampler/data cursor state restore.
   - Phase 3: resume invariant checks and optional next-batch fingerprint.
5. Timebox deeper platform refactors until after core experiment matrix is complete.

---

## 13) Quick executive summary

- We confirmed and fixed a real adapter metadata bug for wrapped discriminative models.
- We confirmed `ancdata` crashes were tied to multiprocess loader paths (often train loader in image logging), not just validation workers.
- We aligned on a more robust recovery philosophy: fail-stop + restart from committed checkpoints.
- We identified that current checkpoint state is model-centric but not yet deterministic-data-stream complete.
- We collected a practical reference set and search vocabulary to implement restart hardening without losing focus on publishable outcomes.

