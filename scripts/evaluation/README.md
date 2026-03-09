# Evaluation Package

Unified evaluation package for:
- slice-level 2D metric aggregation,
- volume-level 3D metric aggregation,
- optional reconstructed volume NIfTI export for QA.

The runtime is streaming-first: slices are consumed in order, and volume metrics
are updated when a volume boundary is reached (no full-dataset volume buffering
required for metric computation).

## Entry points

- Diffusion/custom model:
  - `python3 -m scripts.evaluation.compute_segmentation_metrics_for_diffusionmodel_2d_predictions ...`
- nnU-Net post-threshold predictions:
  - `python3 -m scripts.evaluation.compute_segmentation_metrics_for_nnunet_2d_predictions ...`

## Key flags

### Shared volume-export flags

- `--export-reconstructed-volumes`
  - Writes reconstructed 3D NIfTI per case (`pred` and `gt`).
- `--max-export-volumes-per-case`
  - Optional cap for exported reconstructed volumes.

### Diffusion/custom flags

- Threshold protocol:
  - `--thresholds 0.05:0.95:0.05` (sweep mode)
  - `--fixed-threshold 0.5` (fixed mode)
  - `--optimize-metric dice`
- Ensemble:
  - `--ensemble-samples 1` or `--ensemble-samples 1,3,5`
  - `--ensemble-method single|mean|soft_staple|both`
  - `--staple-max-iters`
  - `--staple-tolerance`
- Quick test:
  - `--test --test-max-slices 10`

### nnU-Net flags

- `--fixed-threshold 0.5`
- `--allow-shape-mismatch`

## Outputs

For each analysis case (or one default case for nnU-Net), outputs include:

- `canonical_results.json`
- `metrics_per_threshold.csv` (slice-level)
- `volume_metrics_per_threshold.csv` (volume-level)
- `evaluation_summary.txt`

If export is enabled:
- `reconstructed_volumes/<analysis_case_key>/<volume_id>__pred.nii.gz`
- `reconstructed_volumes/<analysis_case_key>/<volume_id>__gt.nii.gz`

## Example commands

### Diffusion/custom (local)

```bash
python3 -m scripts.evaluation.compute_segmentation_metrics_for_diffusionmodel_2d_predictions \
  --run-dir /mnt/outputs/<run_dir> \
  --model-name <checkpoint_name_without_pth> \
  --thresholds 0.05:0.95:0.05 \
  --optimize-metric dice \
  --ensemble-samples 3 \
  --ensemble-method both \
  --export-reconstructed-volumes \
  --max-export-volumes-per-case 10
```

### nnU-Net (local)

```bash
python3 -m scripts.evaluation.compute_segmentation_metrics_for_nnunet_2d_predictions \
  --pred-dir /mnt/outputs/nnunet/preds \
  --gt-dir /mnt/outputs/nnunet/labels \
  --fixed-threshold 0.5 \
  --export-reconstructed-volumes
```

### Diffusion/custom via SLURM runner

```bash
python3 -m scripts.evaluation.slurm_runners.run_compute_segmentation_metrics_for_diffusionmodel_2d_predictions \
  --run-dir /mnt/outputs/<run_dir> \
  --model-name <checkpoint_name_without_pth> \
  --thresholds 0.05:0.95:0.05 \
  --optimize-metric dice \
  --ensemble-samples 3 \
  --ensemble-method both \
  --export-reconstructed-volumes \
  --max-export-volumes-per-case 10
```

### nnU-Net via SLURM runner

```bash
python3 -m scripts.evaluation.slurm_runners.run_compute_segmentation_metrics_for_nnunet_2d_predictions \
  --pred-dir /mnt/outputs/nnunet/preds \
  --gt-dir /mnt/outputs/nnunet/labels \
  --fixed-threshold 0.5 \
  --export-reconstructed-volumes
```

