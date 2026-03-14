#!/usr/bin/env python3
"""
Migrate a legacy Hydra run config to the explicit data contract.

Default behavior is in-place rewrite of:
  <run_dir>/.hydra/config.yaml
while preserving a backup of the original file.

This migration targets legacy ISLES24 online 3D->2D runs that predate
the explicit data contract (`data_mode`, `data_io`, `data_runtime`).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import validate_dataset_contract  # noqa: E402


def _is_set(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return len(value.strip()) > 0
    return True


def _select_first_set(cfg: DictConfig, *keys: str) -> Any:
    for key in keys:
        value = OmegaConf.select(cfg, key, default=None)
        if _is_set(value):
            return value
    return None


def _delete_key(cfg: DictConfig, key: str) -> bool:
    if "." not in key:
        if key in cfg:
            del cfg[key]
            return True
        return False

    parent_key, leaf = key.rsplit(".", 1)
    parent = OmegaConf.select(cfg, parent_key, default=None)
    if parent is None:
        return False
    if isinstance(parent, DictConfig) and leaf in parent:
        del parent[leaf]
        return True
    if isinstance(parent, dict) and leaf in parent:
        del parent[leaf]
        return True
    return False


def _migrate_data_runtime(cfg: DictConfig) -> None:
    runtime_mappings = {
        "data_runtime.train_batch_size": (
            "dataset.train_batch_size",
            "environment.dataset.train_batch_size",
        ),
        "data_runtime.test_batch_size": (
            "dataset.test_batch_size",
            "environment.dataset.test_batch_size",
        ),
        "data_runtime.num_train_workers": (
            "dataset.num_train_workers",
            "environment.dataset.num_train_workers",
        ),
        "data_runtime.num_valid_workers": (
            "dataset.num_valid_workers",
            "environment.dataset.num_valid_workers",
        ),
        "data_runtime.num_test_workers": (
            "dataset.num_test_workers",
            "environment.dataset.num_test_workers",
        ),
        "data_runtime.use_caching": ("dataset.use_caching",),
        "data_runtime.use_shared_cache": ("dataset.use_shared_cache",),
        "data_runtime.train_prefetch_factor": ("dataset.train_prefetch_factor",),
        "data_runtime.test_prefetch_factor": ("dataset.test_prefetch_factor",),
    }

    defaults = {
        "data_runtime.use_caching": False,
        "data_runtime.use_shared_cache": False,
        "data_runtime.train_prefetch_factor": 2,
        "data_runtime.test_prefetch_factor": 2,
    }

    missing_required = []
    required_without_defaults = (
        "data_runtime.train_batch_size",
        "data_runtime.test_batch_size",
        "data_runtime.num_train_workers",
        "data_runtime.num_valid_workers",
        "data_runtime.num_test_workers",
    )

    for target_key, legacy_sources in runtime_mappings.items():
        value = _select_first_set(cfg, target_key, *legacy_sources)
        if _is_set(value):
            OmegaConf.update(cfg, target_key, value, merge=False)
            continue

        if target_key in defaults:
            OmegaConf.update(cfg, target_key, defaults[target_key], merge=False)
            continue

        if target_key in required_without_defaults:
            missing_required.append(target_key)

    if missing_required:
        missing_csv = ", ".join(missing_required)
        raise ValueError(
            "Cannot migrate legacy runtime keys. Missing required values for: "
            f"{missing_csv}."
        )

    for split in ("train", "val", "test"):
        pin_key = f"data_runtime.pin_memory.{split}"
        persist_key = f"data_runtime.persistent_workers.{split}"
        if not _is_set(OmegaConf.select(cfg, pin_key, default=None)):
            OmegaConf.update(cfg, pin_key, False, merge=False)
        if not _is_set(OmegaConf.select(cfg, persist_key, default=None)):
            OmegaConf.update(cfg, persist_key, False, merge=False)


def migrate_legacy_online_config(cfg: DictConfig) -> DictConfig:
    OmegaConf.set_struct(cfg, False)

    dataset_name = _select_first_set(cfg, "dataset.name", "dataset.id")
    if not _is_set(dataset_name):
        raise ValueError("Cannot determine dataset identity from dataset.name / dataset.id.")

    dataset_name = str(dataset_name)
    dataset_id = str(_select_first_set(cfg, "dataset.id", "dataset.name") or dataset_name)
    if dataset_id != "isles24":
        raise ValueError(
            "This migrator currently supports legacy ISLES24 online runs only. "
            f"Got dataset.id/name='{dataset_id}'."
        )

    modalities = OmegaConf.select(cfg, "dataset.modalities", default=None)
    modalities_ok = isinstance(modalities, (list, tuple)) or OmegaConf.is_list(modalities)
    if not modalities_ok or len(modalities) == 0:
        raise ValueError("Cannot migrate: dataset.modalities is missing or empty.")

    if not _is_set(OmegaConf.select(cfg, "dataset.num_modalities", default=None)):
        OmegaConf.update(cfg, "dataset.num_modalities", len(modalities), merge=False)

    # Canonical dataset identity and label contract.
    OmegaConf.update(cfg, "dataset.id", dataset_id, merge=False)
    OmegaConf.update(cfg, "dataset.name", dataset_name, merge=False)
    if not _is_set(OmegaConf.select(cfg, "dataset.label_spec.background", default=None)):
        OmegaConf.update(cfg, "dataset.label_spec.background", 0, merge=False)
    if not _is_set(OmegaConf.select(cfg, "dataset.label_spec.foreground", default=None)):
        OmegaConf.update(cfg, "dataset.label_spec.foreground", 1, merge=False)

    # Legacy runs covered by this migration are online 3d->2d.
    existing_loader_mode = OmegaConf.select(cfg, "data_mode.loader_mode", default=None)
    if _is_set(existing_loader_mode) and str(existing_loader_mode) != "online_slices_3d_to_2d":
        raise ValueError(
            "Refusing migration: existing data_mode.loader_mode is not online_slices_3d_to_2d. "
            f"Got '{existing_loader_mode}'."
        )
    OmegaConf.update(cfg, "data_mode.loader_mode", "online_slices_3d_to_2d", merge=False)
    OmegaConf.update(cfg, "data_mode.dim", "2d", merge=False)
    OmegaConf.update(
        cfg,
        "data_mode.preprocessing_mode",
        "online_volume_to_slice_preprocessed",
        merge=False,
    )

    data_root = _select_first_set(
        cfg,
        "data_io.paths.data_root",
        "environment.dataset.data_root",
        "environment.dataset.dir",
    )
    split_file = _select_first_set(
        cfg,
        "data_io.paths.split_file",
        "environment.dataset.split_file",
        "environment.dataset.json_list",
    )
    nnunet_root = _select_first_set(
        cfg,
        "data_io.paths.nnunet_root",
        "environment.dataset.nnunet_root",
    )

    if not _is_set(data_root) or not _is_set(split_file):
        raise ValueError(
            "Cannot migrate: missing legacy dataset paths. Expected one of "
            "environment.dataset.{data_root,dir} and environment.dataset.{split_file,json_list}."
        )

    OmegaConf.update(cfg, "data_io.paths.data_root", data_root, merge=False)
    OmegaConf.update(cfg, "data_io.paths.split_file", split_file, merge=False)
    OmegaConf.update(cfg, "data_io.paths.nnunet_root", nnunet_root, merge=False)

    # Keep environment paths aligned with canonical naming for newer tooling.
    OmegaConf.update(cfg, "environment.dataset.data_root", data_root, merge=False)
    OmegaConf.update(cfg, "environment.dataset.split_file", split_file, merge=False)

    _migrate_data_runtime(cfg)

    if not _is_set(OmegaConf.select(cfg, "distribution.timeout_minutes", default=None)):
        OmegaConf.update(cfg, "distribution.timeout_minutes", 60, merge=False)

    # Remove legacy alias keys from migrated output config.
    for legacy_key in (
        "environment.dataset.dir",
        "environment.dataset.json_list",
        "environment.dataset.train_batch_size",
        "environment.dataset.test_batch_size",
        "environment.dataset.num_train_workers",
        "environment.dataset.num_valid_workers",
        "environment.dataset.num_test_workers",
        "dataset.train_batch_size",
        "dataset.test_batch_size",
        "dataset.num_train_workers",
        "dataset.num_valid_workers",
        "dataset.num_test_workers",
        "dataset.use_caching",
        "dataset.use_shared_cache",
        "dataset.train_prefetch_factor",
        "dataset.test_prefetch_factor",
    ):
        _delete_key(cfg, legacy_key)

    OmegaConf.set_struct(cfg, True)

    # Enforce that the migrated config satisfies current contract checks.
    validate_dataset_contract(cfg)
    return cfg


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Migrate one legacy run config to explicit data contract format.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to run directory containing .hydra/config.yaml",
    )
    parser.add_argument(
        "--config-relative-path",
        type=Path,
        default=Path(".hydra/config.yaml"),
        help="Config path relative to run dir (default: .hydra/config.yaml)",
    )
    parser.add_argument(
        "--backup-filename",
        type=str,
        default="config.legacy_pre_data_contract.yaml",
        help="Backup filename written next to the config before migration.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print summary, but do not write files.",
    )
    parser.add_argument(
        "--overwrite-backup",
        action="store_true",
        help="Allow replacing existing backup file.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    config_path = (run_dir / args.config_relative_path).resolve()
    backup_path = config_path.with_name(args.backup_filename)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"Expected DictConfig at {config_path}, got {type(cfg)}")

    before_yaml = OmegaConf.to_yaml(cfg, resolve=False)
    migrated_cfg = migrate_legacy_online_config(cfg)
    after_yaml = OmegaConf.to_yaml(migrated_cfg, resolve=False)
    changed = before_yaml != after_yaml

    print(f"Run dir: {run_dir}")
    print(f"Config:  {config_path}")
    print(f"Backup:  {backup_path}")
    print(f"Changed: {changed}")

    if args.dry_run:
        print("Dry-run complete. No files written.")
        return

    if backup_path.exists() and not args.overwrite_backup:
        raise FileExistsError(
            "Backup already exists. Use --overwrite-backup to replace it: "
            f"{backup_path}"
        )

    shutil.copy2(config_path, backup_path)
    OmegaConf.save(config=migrated_cfg, f=str(config_path), resolve=False)
    print("Migration applied successfully.")


if __name__ == "__main__":
    main()

