#!/usr/bin/env python3
"""
Convert nnU-Net datasets from config-driven data loader routes.

Supported conversion modes:
    - data_mode.loader_mode = online_slices_3d_to_2d (2D slice export)
    - data_mode.loader_mode = full_volumes_3d (3D volume export)

Supported dataset routes currently include ISLES24/ISLES26 through existing
loader stack contracts.

Usage:
    # Local environment - test mode (default - processes limited samples)
    python3 -m scripts.nnunet.convert_to_nnunet \
        --config-name=nnunet/convert/isles24_local

    # Cluster environment
    python3 -m scripts.nnunet.convert_to_nnunet \
        --config-name=nnunet/convert/isles24_cluster_baseline

    # ISLES26 (local, T1 raw)
    python3 -m scripts.nnunet.convert_to_nnunet \
        --config-name=nnunet/convert/isles26_local_t1raw

    # ISLES26 (cluster, T1 raw)
    python3 -m scripts.nnunet.convert_to_nnunet \
        --config-name=nnunet/convert/isles26_cluster_t1raw

    # ISLES24 3D (local, baseline modalities)
    python3 -m scripts.nnunet.convert_to_nnunet \
        --config-name=nnunet/convert/isles24_local_3d_baseline

    # ISLES26 3D (cluster, T1 raw)
    python3 -m scripts.nnunet.convert_to_nnunet \
        --config-name=nnunet/convert/isles26_cluster_3d_t1raw

    # Full export
    python3 -m scripts.nnunet.convert_to_nnunet \
        --config-name=nnunet/convert/isles24_local nnunet.test=false

    # Override output location
    python3 -m scripts.nnunet.convert_to_nnunet \
        --config-name=nnunet/convert/isles24_local \
        nnunet.output_dir=/mnt/data/nnUNet_raw

    # Different fold
    python3 -m scripts.nnunet.convert_to_nnunet \
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
    Public helper that validates converter contract requirements.
    """
    _validate_converter_contract(cfg)


def _build_export_affine(
    slice_meta: Dict[str, Any],
    out_h: int,
    out_w: int,
    slice_idx: int,
) -> Any:
    """
    Helper around slice affine construction used by converter tests.
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
    Helper around single-slice export used by converter tests.
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
    Helper around sequential split export used by converter tests/tooling.
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
    Helper around parallel split export used by converter tests/tooling.
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
    Helper for provenance JSONL writing used by converter tests/tooling.
    """
    _write_provenance_jsonl(records=records, output_path=output_path)


@hydra.main(config_path="../../configs", config_name="nnunet/convert/isles24_local", version_base=None)
def main(cfg: DictConfig):
    """
    Main conversion entrypoint.

    Reuses existing config infrastructure and data loading.
    All settings come from config - no defaults in code.
    """
    run_conversion(cfg)


if __name__ == "__main__":
    main()
