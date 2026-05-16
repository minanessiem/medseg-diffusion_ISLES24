#!/usr/bin/env python3
"""
Convert 2D online-slice datasets to nnU-Net v2 format.

This script currently supports ISLES24/ISLES26 online 3D->2D routes loaded via
get_dataloaders(). It exports each slice to nnU-Net-compatible 2D NIfTI files
for benchmarking against diffusion-based segmentation methods.

Dataset assumptions:
    - Uses data_mode.loader_mode = online_slices_3d_to_2d
    - Parses virtual paths in format "{caseID}_slice{idx}"
    - Outputs 2D NIfTI files with shape [H, W, 1]

For full-volume 3D export routes, use the generic conversion entrypoint once
introduced by the nnU-Net pipeline refactor.

Usage:
    # Local environment - test mode (default - processes limited slices)
    python3 -m scripts.nnunet.convert_isles24_2d_dataset_to_nnunet \
        --config-name=nnunet/convert/isles24_local
    
    # Cluster environment
    python3 -m scripts.nnunet.convert_isles24_2d_dataset_to_nnunet \
        --config-name=nnunet/convert/isles24_cluster_baseline

    # ISLES26 (local, T1 raw)
    python3 -m scripts.nnunet.convert_isles24_2d_dataset_to_nnunet \
        --config-name=nnunet/convert/isles26_local_t1raw

    # ISLES26 (cluster, T1 raw)
    python3 -m scripts.nnunet.convert_isles24_2d_dataset_to_nnunet \
        --config-name=nnunet/convert/isles26_cluster_t1raw
    
    # Full export
    python3 -m scripts.nnunet.convert_isles24_2d_dataset_to_nnunet \
        --config-name=nnunet/convert/isles24_local nnunet.test=false
    
    # Override output location
    python3 -m scripts.nnunet.convert_isles24_2d_dataset_to_nnunet \
        --config-name=nnunet/convert/isles24_local \
        nnunet.output_dir=/mnt/data/nnUNet_raw
    
    # Different fold
    python3 -m scripts.nnunet.convert_isles24_2d_dataset_to_nnunet \
        --config-name=nnunet/convert/isles24_local \
        dataset.fold=2
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Import MONAI first (before CUDA context creation)
try:
    from monai.transforms import Resize, ScaleIntensityRange
except ImportError:
    pass

import hydra
from omegaconf import DictConfig
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.nnunet.core.conversion_core import (
    run_conversion,
    validate_converter_contract as _validate_converter_contract,
)
from scripts.nnunet.core.exporters import (
    SliceExportStrategy,
    write_provenance_jsonl as _write_provenance_jsonl,
)

_SLICE_EXPORTER = SliceExportStrategy()


def validate_converter_contract(cfg: DictConfig) -> None:
    """
    Backward-compatible wrapper for core conversion contract validation.
    """
    _validate_converter_contract(cfg)


def _build_export_affine(
    slice_meta: Dict[str, Any],
    out_h: int,
    out_w: int,
    slice_idx: int,
) -> Any:
    """
    Backward-compatible wrapper around core slice affine construction.
    """
    return _SLICE_EXPORTER._build_export_affine(
        slice_meta=slice_meta,
        out_h=out_h,
        out_w=out_w,
        slice_idx=slice_idx,
    )


def process_single_slice(
    idx: int,
    dataset: Any,
    images_dir: Path,
    labels_dir: Path,
    num_channels: int,
    split_name: str,
) -> Dict[str, Any]:
    """
    Backward-compatible wrapper around core slice export processing.
    """
    return _SLICE_EXPORTER._process_single_slice(
        idx=idx,
        dataset=dataset,
        images_dir=images_dir,
        labels_dir=labels_dir,
        num_channels=num_channels,
        split_name=split_name,
    )


def export_dataset(
    dataset: Any,
    images_dir: Path,
    labels_dir: Path,
    num_channels: int,
    desc: str,
    max_slices: Optional[int] = None,
    split_name: str = "unknown",
) -> Tuple[set, List[Dict[str, Any]]]:
    """
    Backward-compatible wrapper around sequential split export.
    """
    return _SLICE_EXPORTER._export_dataset_sequential(
        dataset=dataset,
        images_dir=images_dir,
        labels_dir=labels_dir,
        num_channels=num_channels,
        desc=desc,
        max_slices=max_slices,
        split_name=split_name,
    )


def export_dataset_parallel(
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
    Backward-compatible wrapper around parallel split export.
    """
    return _SLICE_EXPORTER._export_dataset_parallel(
        dataset=dataset,
        images_dir=images_dir,
        labels_dir=labels_dir,
        num_channels=num_channels,
        desc=desc,
        max_slices=max_slices,
        num_workers=num_workers,
        split_name=split_name,
    )


def write_provenance_jsonl(records: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Backward-compatible wrapper for provenance JSONL writing.
    """
    _write_provenance_jsonl(records=records, output_path=output_path)


@hydra.main(config_path="../../configs", config_name="nnunet/convert/isles24_local", version_base=None)
def main(cfg: DictConfig):
    """
    Main conversion entry point.
    
    Reuses existing config infrastructure and data loading.
    All settings come from config - no defaults in code.
    """
    # Phase 2: thin entrypoint delegating to extracted conversion core.
    run_conversion(cfg)

if __name__ == "__main__":
    main()

