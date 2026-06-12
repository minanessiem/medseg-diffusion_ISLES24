# Analysis Reproducibility Guide

This document is the "come back in 2 months and run it all again" command index for
the analysis pipeline.

Run all commands from the repository root:

```bash
cd c:\Users\minanessiem\Development\medseg-diffusion
```

---

## 1) Inputs You Need

- `RUN_DIR`: training run folder containing `.hydra/config.yaml` and checkpoints
- `MODEL_NAME`: checkpoint name without `.pth`
- `NNUNET_PRED_DIR`: folder with nnUNet prediction volumes (`*.nii.gz`) for paper grids

Example placeholders used below:

- `<RUN_DIR>` e.g. `outputs/discriminative_swinunetr_.../run_...`
- `<MODEL_NAME>` e.g. `best_model_step_...`
- `<NNUNET_PRED_DIR>` e.g. `nnUNet_results/predictionsTs`

---

## 2) Model Evaluation and Thresholding (Local)

Use the config-driven evaluation entrypoint for new repository-model threshold
calibration, especially ISLES26 3D discriminative volume evaluation:

```bash
python3 -m scripts.evaluation.evaluate_model --help
```

### A) Fixed threshold at the saved run config

```bash
python3 -m scripts.evaluation.evaluate_model evaluation.run_dir=<RUN_DIR> evaluation.model_name=<MODEL_NAME>
```

### B) 3D volume threshold sweep with per-case oracle

```bash
python3 -m scripts.evaluation.evaluate_model --evaluation-config-name threshold_sweep_with_oracle evaluation.run_dir=<RUN_DIR> evaluation.model_name=<MODEL_NAME> dataset.active_subsets.val=val_full validation=sliding_window_3d_metrics_full validation.val_batch_size=1 evaluation.threshold_protocol.primary.level=volume evaluation.threshold_protocol.primary.metric=DiceNativeCoefficient
```

Default output:

- `<RUN_DIR>/analysis/evaluation_v3/<MODEL_NAME>_<timestamp>/`

Key files:

- `canonical_results.json`
- `evaluation_summary.txt`
- `resolved_evaluation_config.yaml`
- `volume_metrics_per_threshold.csv`
- `per_case_threshold_metrics.csv`
- `oracle_per_case_thresholds.csv` when oracle mode is enabled

---

## 3) Legacy Threshold Analysis (Local)

`scripts.analysis.threshold_analysis` is the legacy threshold-analysis path. Keep
using it when you need its plotting or older 2D diffusion ensemble workflows.
For new config-driven repository-model evaluation and 3D volume thresholding,
prefer `python3 -m scripts.evaluation.evaluate_model`.

Main script:

```bash
python3 -m scripts.analysis.threshold_analysis --help
```

### A) Paper-style primary protocol (optimize Dice, report all metrics at selected tau)

```bash
python3 -m scripts.analysis.threshold_analysis --run-dir <RUN_DIR> --model-name <MODEL_NAME> --optimize-metric dice --thresholds 0.05:0.95:0.05 --viz-selection random_seeded --viz-seed 42 --num-visualizations 6
```

### B) Fixed-threshold protocol (no optimization)

```bash
python3 -m scripts.analysis.threshold_analysis --run-dir <RUN_DIR> --model-name <MODEL_NAME> --fixed-threshold 0.50 --viz-selection random_seeded --viz-seed 42 --num-visualizations 6
```

### C) Diffusion ensemble sweep in one run (1/3/5, both merge methods)

```bash
python3 -m scripts.analysis.threshold_analysis --run-dir <RUN_DIR> --model-name <MODEL_NAME> --optimize-metric dice --thresholds 0.05:0.95:0.05 --ensemble-samples 1,3,5 --ensemble-method both --staple-max-iters 5 --staple-tolerance 0.02 --viz-selection random_seeded --viz-seed 42 --num-visualizations 6
```

### D) Deterministic fixed visualization slices

```bash
python3 -m scripts.analysis.threshold_analysis --run-dir <RUN_DIR> --model-name <MODEL_NAME> --optimize-metric dice --thresholds 0.05:0.95:0.05 --viz-selection fixed_indices --viz-indices 12,44,87,103 --num-visualizations 4
```

### E) Exploratory-only top-improvement visualizations

```bash
python3 -m scripts.analysis.threshold_analysis --run-dir <RUN_DIR> --model-name <MODEL_NAME> --optimize-metric dice --thresholds 0.05:0.95:0.05 --viz-selection top_improvement --num-visualizations 4
```

---

## 4) Legacy Threshold Analysis (SLURM)

The SLURM runner below targets the legacy analysis script. A SLURM runner for
`scripts.evaluation.evaluate_model` is intentionally deferred until the local
CLI has been validated on real runs.

Runner script:

```bash
python3 -m scripts.analysis.slurm_runners.run_threshold_analysis --help
```

### A) Dry-run first (recommended)

```bash
python3 -m scripts.analysis.slurm_runners.run_threshold_analysis --run-dir <RUN_DIR> --model-name <MODEL_NAME> --optimize-metric dice --thresholds 0.05:0.95:0.05 --viz-selection random_seeded --viz-seed 42 --num-visualizations 6 --gpus 1 --cpus-per-task 16 --mem 64G --time 00:45:00 --dry-run
```

### B) Submit

```bash
python3 -m scripts.analysis.slurm_runners.run_threshold_analysis --run-dir <RUN_DIR> --model-name <MODEL_NAME> --optimize-metric dice --thresholds 0.05:0.95:0.05 --viz-selection random_seeded --viz-seed 42 --num-visualizations 6 --gpus 1 --cpus-per-task 16 --mem 64G --time 00:45:00
```

Note: the current SLURM runner accepts a single integer for `--ensemble-samples`.
If you need comma-separated multi-case ensemble analysis (`1,3,5`) use the local
`threshold_analysis.py` command directly.

---

## 5) Paper Comparison Grid (MedSegDiff 1/3/5 + nnUNet)

Main script:

```bash
python3 -m scripts.analysis.paper_comparison_grid --help
```

### A) Generate fresh figures

```bash
python3 -m scripts.analysis.paper_comparison_grid --run-dir <RUN_DIR> --model-name <MODEL_NAME> --nnunet-pred-dir <NNUNET_PRED_DIR> --num-visualizations 6 --viz-seed 42 --prediction-threshold 0.50 --foreground-ratio 0.25 --max-selection-checks 256 --ensemble-method mean --save-pdf
```

### B) Re-render from cache (faster reruns, no inference)

```bash
python3 -m scripts.analysis.paper_comparison_grid --run-dir <RUN_DIR> --model-name <MODEL_NAME> --nnunet-pred-dir <NNUNET_PRED_DIR> --use-cached --cache-dir <CACHE_DIR> --font-size 30 --prediction-threshold 0.50
```

### C) Deterministic fixed slices (paper curation)

```bash
python3 -m scripts.analysis.paper_comparison_grid --run-dir <RUN_DIR> --model-name <MODEL_NAME> --nnunet-pred-dir <NNUNET_PRED_DIR> --viz-indices 12,44,87,103 --num-visualizations 4 --prediction-threshold 0.50 --foreground-ratio 0.25 --ensemble-method mean
```

### D) Notes from implementation/testing

- Run with module form (`python3 -m scripts.analysis.paper_comparison_grid`), not file path execution.
- `--prediction-threshold` controls MedSegDiff binarization in rendered panels (`tau` in paper text).
- `--foreground-ratio` filters slice eligibility by GT foreground fraction (default `0.25`).
- `--max-selection-checks` bounds the selection scan to avoid long “hang-like” scans.
- The script renders flush tiles (no spacing), creates `comparison_rows_stacked.png`, draws bars, bottom-row column labels, and per-row case/slice headers.
- `--font-size` now overrides logger config `label_font_size` for stacked-image labels.

---

## 6) Where Outputs Go

### Config-driven model evaluation default output

- `<RUN_DIR>/analysis/evaluation_v3/<MODEL_NAME>_<timestamp>/`

Key files:

- `canonical_results.json`
- `evaluation_summary.txt`
- `resolved_evaluation_config.yaml`
- `volume_metrics_per_threshold.csv`
- `per_case_threshold_metrics.csv`
- `oracle_per_case_thresholds.csv` when oracle mode is enabled

### Legacy threshold analysis default output

- `<RUN_DIR>/analysis/threshold_sweep/<MODEL_NAME>_<timestamp>/`

Key files:

- `*_summary.txt`
- `*_optimal_thresholds.json`
- `*_metrics_per_threshold.csv`
- `*_metrics_per_sample.csv` (only with `--save-per-sample`)
- `plots/`
- `visualizations/`

When multiple ensemble cases are analyzed in one run, output files are prefixed
per case, for example:

- `n1_single_summary.txt`
- `n3_mean_summary.txt`
- `n3_soft_staple_summary.txt`
- `n5_mean_summary.txt`
- `n5_soft_staple_summary.txt`

### Paper comparison grid default output

- `<RUN_DIR>/analysis/paper_comparison/<MODEL_NAME>_<timestamp>/`
- Stable cache root: `<RUN_DIR>/analysis/paper_comparison/cache/`
- Default cache key folder (auto): `<RUN_DIR>/analysis/paper_comparison/cache/<model+selection key>/`

Key files:

- `figures/*.png` (and `*.pdf` when `--save-pdf`)
- `comparison_rows_stacked.png`
- `manifest.json`
- `manifest.csv`
- `run_metadata.json`
- cache payloads under the cache key folder:
  - `render_arrays.npz`
  - `render_metadata.json`

---

## 7) Minimal "Rebuild Paper Numbers" Workflow

1. For new 3D repository-model numbers, run `scripts.evaluation.evaluate_model`
   with an explicit fixed or sweep-with-oracle threshold protocol.
2. For legacy 2D/plotting workflows, run threshold analysis with your paper
   protocol (`optimize-metric dice` or fixed tau).
3. Read the corresponding summary/report JSON files for table values.
4. Run `paper_comparison_grid.py` for figure assets.
5. Keep command + output folder in your notes/manuscript appendix.

Recommended to lock reproducibility:

- use `--viz-selection random_seeded --viz-seed 42` (or `fixed_indices`)
- keep threshold protocol explicit in methods (`optimized_metric` vs `fixed_threshold`)
- keep same-split caveat (fold-0 only) in limitations

---

## 8) Quick Sanity Commands

Check scripts still parse:

```bash
python3 -m scripts.evaluation.evaluate_model --help
python3 -m scripts.analysis.threshold_analysis --help
python3 -m scripts.analysis.slurm_runners.run_threshold_analysis --help
python3 -m scripts.analysis.paper_comparison_grid --help
```

Syntax checks:

```bash
python3 -m py_compile scripts/evaluation/evaluate_model.py
python3 -m py_compile scripts/analysis/threshold_analysis.py
python3 -m py_compile scripts/analysis/slurm_runners/run_threshold_analysis.py
python3 -m py_compile scripts/analysis/paper_comparison_grid.py
```

