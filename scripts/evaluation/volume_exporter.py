"""
Export reconstructed channel-first volumes to NIfTI for visual QA.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np

from scripts.evaluation.contracts import VolumeSample


def export_reconstructed_volumes(
    grouped_volumes: Dict[str, List[VolumeSample]],
    output_dir: Path,
    max_volumes_per_case: Optional[int] = None,
) -> List[Path]:
    """
    Export reconstructed prediction/GT volumes to NIfTI files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for case_key in sorted(grouped_volumes.keys()):
        case_dir = output_dir / str(case_key)
        case_dir.mkdir(parents=True, exist_ok=True)
        volumes = grouped_volumes[case_key]
        if max_volumes_per_case is not None and max_volumes_per_case >= 0:
            volumes = volumes[: max_volumes_per_case]
        for volume_sample in volumes:
            volume_sample.validate()
            affine = _resolve_export_affine(volume_sample)
            pred_data = _to_nifti_array(volume_sample.prediction_volume)
            gt_data = _to_nifti_array(volume_sample.ground_truth_volume)
            pred_path = case_dir / f"{volume_sample.volume_id}__pred.nii.gz"
            gt_path = case_dir / f"{volume_sample.volume_id}__gt.nii.gz"
            _write_nifti(pred_data, affine, pred_path)
            _write_nifti(gt_data, affine, gt_path)
            written.extend([pred_path, gt_path])
    return written


def _resolve_export_affine(volume_sample: VolumeSample) -> np.ndarray:
    """
    Build export affine from first-slice metadata when available.
    """
    first_meta = dict(volume_sample.metadata.get("first_slice_metadata", {}))
    raw_affine = first_meta.get("source_affine")
    if raw_affine is None:
        return np.eye(4, dtype=np.float64)
    source_affine = np.asarray(raw_affine, dtype=np.float64)
    if source_affine.shape != (4, 4):
        return np.eye(4, dtype=np.float64)
    pre_hw = first_meta.get("pre_resize_shape_hw")
    if pre_hw is None:
        pre_hw = [int(volume_sample.prediction_volume.shape[1]), int(volume_sample.prediction_volume.shape[2])]
    out_h = int(volume_sample.prediction_volume.shape[1])
    out_w = int(volume_sample.prediction_volume.shape[2])
    pre_h = max(int(pre_hw[0]), 1)
    pre_w = max(int(pre_hw[1]), 1)
    scale_h = float(pre_h / max(out_h, 1))
    scale_w = float(pre_w / max(out_w, 1))
    slice_indices = volume_sample.metadata.get("slice_indices", [0])
    min_slice_idx = int(min(slice_indices)) if slice_indices else 0

    export_affine = np.array(source_affine, dtype=np.float64, copy=True)
    export_affine[:3, 0] = source_affine[:3, 0] * scale_h
    export_affine[:3, 1] = source_affine[:3, 1] * scale_w
    export_affine[:3, 3] = source_affine[:3, 3] + source_affine[:3, 2] * float(min_slice_idx)
    return export_affine


def _to_nifti_array(volume_chwd) -> np.ndarray:
    # [C,H,W,D] -> [H,W,D] (single-channel expected)
    arr = volume_chwd.detach().cpu().numpy()
    if arr.ndim != 4:
        raise ValueError(f"Expected [C,H,W,D], got {arr.shape}")
    if arr.shape[0] != 1:
        raise ValueError(f"Expected single-channel volume, got C={arr.shape[0]}")
    return arr[0].astype(np.float32)


def _write_nifti(data_hwd: np.ndarray, affine: np.ndarray, path: Path) -> None:
    nii = nib.Nifti1Image(data_hwd, affine=affine)
    nii.set_qform(affine, code=1)
    nii.set_sform(affine, code=1)
    nib.save(nii, path)
