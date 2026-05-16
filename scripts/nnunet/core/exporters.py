"""
Export strategies and low-level filesystem helpers for nnU-Net conversion.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from tqdm import tqdm


def clear_directory(directory: Path, description: str = "") -> int:
    """
    Remove all files from a directory, keeping the directory itself.

    Args:
        directory: Path to directory to clear.
        description: Human-readable description for logging.

    Returns:
        Number of files removed.
    """
    if not directory.exists():
        return 0

    files = list(directory.glob("*"))
    count = len(files)

    if count > 0:
        # Remove and recreate for speed (faster than individual deletes)
        shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
        label = f" ({description})" if description else ""
        print(f"  Cleared {count} stale files from {directory.name}/{label}")

    return count


def count_files_in_directory(directory: Path, pattern: str = "*.nii.gz") -> int:
    """
    Count files matching a pattern in a directory.

    Used to determine numTraining from existing files on disk
    when only exporting the test split (resume scenario).

    Args:
        directory: Path to directory.
        pattern: Glob pattern to match.

    Returns:
        Number of matching files.
    """
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def write_provenance_jsonl(records: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Write one JSON record per line for exported slice provenance.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")


class SliceExportStrategy:
    """
    Export strategy for online 3D-to-2D slice datasets.
    """

    @staticmethod
    def _build_export_affine(
        slice_meta: Dict[str, Any],
        out_h: int,
        out_w: int,
        slice_idx: int,
    ) -> np.ndarray:
        """
        Build slice-export affine that preserves source geometry and includes
        index-aware z-offset.
        """
        source_affine = np.asarray(slice_meta.get("source_affine", np.eye(4)), dtype=np.float64)
        if source_affine.shape != (4, 4):
            source_affine = np.eye(4, dtype=np.float64)
        pre_hw = slice_meta.get("pre_resize_shape_hw", [out_h, out_w])
        try:
            pre_h = max(int(pre_hw[0]), 1)
            pre_w = max(int(pre_hw[1]), 1)
        except Exception:
            pre_h, pre_w = out_h, out_w
        out_h = max(int(out_h), 1)
        out_w = max(int(out_w), 1)
        scale_h = float(pre_h / out_h)
        scale_w = float(pre_w / out_w)

        export_affine = np.array(source_affine, dtype=np.float64, copy=True)
        export_affine[:3, 0] = source_affine[:3, 0] * scale_h
        export_affine[:3, 1] = source_affine[:3, 1] * scale_w
        export_affine[:3, 3] = source_affine[:3, 3] + source_affine[:3, 2] * float(slice_idx)
        return export_affine

    @staticmethod
    def _write_nifti_with_affine(data: np.ndarray, affine: np.ndarray, output_path: Path) -> None:
        nii_img = nib.Nifti1Image(data, affine=affine)
        nii_img.set_qform(affine, code=1)
        nii_img.set_sform(affine, code=1)
        nib.save(nii_img, output_path)

    def _process_single_slice(
        self,
        idx: int,
        dataset: Any,
        images_dir: Path,
        labels_dir: Path,
        num_channels: int,
        split_name: str,
    ) -> Dict[str, Any]:
        """
        Process and save a single slice to nnU-Net format.

        Returns:
            Slice export record for provenance manifest.
        """
        sample = dataset[idx]
        if len(sample) == 4:
            image, label, virtual_path, slice_meta = sample
        else:
            image, label, virtual_path = sample
            slice_meta = {}
        # image: [C, H, W], label: [1, H, W]

        # Parse case_id and slice_idx from virtual_path
        # Format from dataset route: "{caseID}_slice{slice_idx}"
        case_id, slice_part = virtual_path.rsplit("_slice", 1)
        slice_idx = int(slice_part)

        # Create case identifier for nnU-Net
        safe_case_id = f"{case_id}_s{slice_idx:04d}"
        out_h = int(image.shape[-2])
        out_w = int(image.shape[-1])
        export_affine = self._build_export_affine(
            slice_meta,
            out_h=out_h,
            out_w=out_w,
            slice_idx=slice_idx,
        )
        image_files: List[str] = []

        # Save each channel as separate file
        for ch_idx in range(num_channels):
            ch_data = image[ch_idx].numpy()  # [H, W]

            # Create 2D NIfTI (shape [H, W, 1] for 2D)
            nii_data = ch_data[..., np.newaxis].astype(np.float32)
            filename = f"{safe_case_id}_{ch_idx:04d}.nii.gz"
            self._write_nifti_with_affine(nii_data, export_affine, images_dir / filename)
            image_files.append(filename)

        # Save label (as uint8 for segmentation)
        label_data = label[0].numpy()  # [H, W]
        label_nii = label_data[..., np.newaxis].astype(np.uint8)
        label_filename = f"{safe_case_id}.nii.gz"
        self._write_nifti_with_affine(label_nii, export_affine, labels_dir / label_filename)

        return {
            "split": split_name,
            "safe_case_id": safe_case_id,
            "case_id": str(case_id),
            "slice_index": int(slice_idx),
            "virtual_path": str(virtual_path),
            "image_files": image_files,
            "label_file": label_filename,
            "source_path": str(slice_meta.get("source_path", "")),
            "source_affine": slice_meta.get("source_affine", np.eye(4).tolist()),
            "source_spacing_xyz": slice_meta.get("source_spacing_xyz", [1.0, 1.0, 1.0]),
            "source_axcodes": slice_meta.get("source_axcodes", ["R", "A", "S"]),
            "source_volume_shape": slice_meta.get("source_volume_shape", [out_h, out_w, 1]),
            "slice_axis": int(slice_meta.get("slice_axis", 2)),
            "pre_resize_shape_hw": slice_meta.get("pre_resize_shape_hw", [out_h, out_w]),
            "post_resize_shape_hw": [out_h, out_w],
            "export_affine": export_affine.tolist(),
        }

    def _export_dataset_sequential(
        self,
        dataset: Any,
        images_dir: Path,
        labels_dir: Path,
        num_channels: int,
        desc: str,
        max_slices: Optional[int] = None,
        split_name: str = "unknown",
    ) -> Tuple[set, List[Dict[str, Any]]]:
        """
        Export a dataset to nnU-Net format sequentially.
        """
        case_ids = set()
        records: List[Dict[str, Any]] = []

        # Determine number of slices to process
        total_slices = len(dataset)
        num_to_process = min(max_slices, total_slices) if max_slices else total_slices

        # Sequential iteration by index
        for idx in tqdm(range(num_to_process), desc=desc):
            record = self._process_single_slice(
                idx,
                dataset,
                images_dir,
                labels_dir,
                num_channels,
                split_name=split_name,
            )
            case_ids.add(record["safe_case_id"])
            records.append(record)

        return case_ids, records

    def _export_dataset_parallel(
        self,
        dataset: Any,
        images_dir: Path,
        labels_dir: Path,
        num_channels: int,
        desc: str,
        max_slices: Optional[int] = None,
        num_workers: int = 32,
        split_name: str = "unknown",
    ) -> Tuple[set, List[Dict[str, Any]]]:
        """
        Export a dataset to nnU-Net format using parallel threads.

        Uses ThreadPoolExecutor for parallel processing. GIL is released
        during I/O operations (nibabel read/write), allowing good parallelism.
        """
        # Determine number of slices to process
        total_slices = len(dataset)
        num_to_process = min(max_slices, total_slices) if max_slices else total_slices

        case_ids = set()
        records: List[Dict[str, Any]] = []

        # Create partial function with fixed arguments
        process_fn = partial(
            self._process_single_slice,
            dataset=dataset,
            images_dir=images_dir,
            labels_dir=labels_dir,
            num_channels=num_channels,
            split_name=split_name,
        )

        # Process slices in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(process_fn, idx): idx for idx in range(num_to_process)}

            # Collect results with progress bar
            for future in tqdm(as_completed(futures), total=num_to_process, desc=desc):
                record = future.result()
                case_ids.add(record["safe_case_id"])
                records.append(record)

        return case_ids, records

    def export_split(
        self,
        dataset: Any,
        images_dir: Path,
        labels_dir: Path,
        num_channels: int,
        desc: str,
        max_slices: Optional[int],
        split_name: str,
        use_parallel: bool,
        num_workers: int,
    ) -> Tuple[set, List[Dict[str, Any]]]:
        """
        Export a split using either sequential or parallel execution.
        """
        if use_parallel:
            return self._export_dataset_parallel(
                dataset=dataset,
                images_dir=images_dir,
                labels_dir=labels_dir,
                num_channels=num_channels,
                desc=desc,
                max_slices=max_slices,
                num_workers=num_workers,
                split_name=split_name,
            )
        return self._export_dataset_sequential(
            dataset=dataset,
            images_dir=images_dir,
            labels_dir=labels_dir,
            num_channels=num_channels,
            desc=desc,
            max_slices=max_slices,
            split_name=split_name,
        )
