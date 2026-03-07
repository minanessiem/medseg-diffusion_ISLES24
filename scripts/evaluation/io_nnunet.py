"""
Streaming IO producer for nnU-Net post-threshold 2D mask predictions.
"""

from pathlib import Path
from typing import Dict, Iterator, Tuple

import nibabel as nib
import torch
from torch import Tensor

from scripts.evaluation.contracts import SliceSample
from scripts.evaluation.mask_builder import build_ground_truth_mask, build_prediction_mask


def iter_nnunet_slice_samples(
    pred_dir: Path,
    gt_dir: Path,
    strict_shape: bool = True,
) -> Iterator[SliceSample]:
    """
    Yield nnU-Net prediction/ground-truth pairs as streaming `SliceSample` objects.

    Each file is treated as a single 2D slice sample and loaded on demand.
    """
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory does not exist: {pred_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth directory does not exist: {gt_dir}")

    gt_files = sorted(gt_dir.glob("*.nii.gz"))
    if not gt_files:
        raise FileNotFoundError(f"No .nii.gz files found in ground truth directory: {gt_dir}")

    pred_lookup: Dict[str, Path] = {
        path.name.replace(".nii.gz", ""): path for path in pred_dir.glob("*.nii.gz")
    }
    if not pred_lookup:
        raise FileNotFoundError(f"No .nii.gz files found in prediction directory: {pred_dir}")

    for gt_file in gt_files:
        case_id = gt_file.name.replace(".nii.gz", "")
        pred_file = pred_lookup.get(case_id)
        if pred_file is None:
            continue

        gt_tensor = _load_nifti_slice_tensor(gt_file)
        pred_tensor = _load_nifti_slice_tensor(pred_file)

        if gt_tensor.shape != pred_tensor.shape:
            if strict_shape:
                raise ValueError(
                    f"Shape mismatch for {case_id}: pred={pred_tensor.shape}, gt={gt_tensor.shape}"
                )
            continue

        yield SliceSample(
            case_id=case_id,
            slice_id=case_id,
            prediction_mask=build_prediction_mask(prediction_mask=pred_tensor),
            ground_truth_mask=build_ground_truth_mask(gt_tensor),
            metadata={
                "source": "nnunet_post_threshold",
                "pred_path": str(pred_file),
                "gt_path": str(gt_file),
            },
        )


def _load_nifti_slice_tensor(path: Path) -> Tensor:
    """
    Load a NIfTI file and return shape [1, H, W].

    Supported input shapes:
    - [H, W]
    - [H, W, 1]
    - [1, H, W]
    """
    data = nib.load(path).get_fdata()
    tensor = torch.from_numpy(data).float()
    return _to_channel_first_slice(tensor, path=path)


def _to_channel_first_slice(tensor: Tensor, path: Path) -> Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    if tensor.ndim == 3 and tensor.shape[-1] == 1:
        return tensor.squeeze(-1).unsqueeze(0)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        return tensor
    raise ValueError(
        f"Unsupported NIfTI shape for {path}: {tuple(tensor.shape)}. "
        "Expected [H,W], [H,W,1], or [1,H,W]."
    )


def count_matched_pairs(pred_dir: Path, gt_dir: Path) -> Tuple[int, int, int]:
    """
    Return (matched_count, missing_prediction_count, total_gt_files).
    """
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    gt_files = sorted(gt_dir.glob("*.nii.gz"))
    pred_lookup = {path.name.replace(".nii.gz", ""): path for path in pred_dir.glob("*.nii.gz")}

    missing = 0
    matched = 0
    for gt_file in gt_files:
        case_id = gt_file.name.replace(".nii.gz", "")
        if case_id in pred_lookup:
            matched += 1
        else:
            missing += 1
    return matched, missing, len(gt_files)

