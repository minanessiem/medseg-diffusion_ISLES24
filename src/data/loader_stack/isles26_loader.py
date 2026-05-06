"""
ISLES26 dataset loaders and parsing helpers.

Phase 4 note:
- Keep structure parallel to `src.data.loader_stack.isles24_loader`.
- Implement datalist parsing plus 3D/2D online dataset foundations.
- nnUNet-specific ISLES26 loader integration is implemented in later phases.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import nibabel
import numpy as np
import torch
from monai.transforms import Resize

ISLES26_MODALITY_KEY = "T1"
ISLES26_SUPPORTED_MODALITY_KEYS = (ISLES26_MODALITY_KEY,)
ISLES26_REQUIRED_RECORD_KEYS = ("fold", "caseID", ISLES26_MODALITY_KEY, "label")
ISLES26_OPTIONAL_RECORD_KEYS = ("siteID", "metadata_csv", "metadata")
ISLES26_VIRTUAL_PATH_TEMPLATE = "{case_id}_slice{slice_idx}"


def _is_non_empty(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return len(value.strip()) > 0
    return True


def _normalize_modalities(modalities: Optional[Sequence[str]]) -> Tuple[str, ...]:
    """
    Normalize modality contract for ISLES26.

    Defaults to the single-modality `T1` channel when no explicit list is given.
    """
    if modalities is None:
        return (ISLES26_MODALITY_KEY,)
    normalized = tuple(str(modality) for modality in modalities)
    if len(normalized) == 0:
        raise ValueError("ISLES26 loaders require a non-empty modalities list.")
    for modality in normalized:
        base_modality = modality.split("_")[0]
        if base_modality != ISLES26_MODALITY_KEY:
            raise ValueError(
                "ISLES26 loaders currently support only T1-derived modality tokens. "
                f"Got modality '{modality}'."
            )
    return normalized


def _resolve_path_value(basedir: str, value: object) -> object:
    """
    Resolve relative datalist paths against `basedir`.
    """
    if isinstance(value, str) and _is_non_empty(value):
        if os.path.isabs(value):
            return value
        return os.path.join(basedir, value)
    if isinstance(value, list):
        return [_resolve_path_value(basedir, item) for item in value]
    return value


def _normalize_case_record_paths(record: Mapping[str, Any], basedir: str) -> Dict[str, Any]:
    """
    Return a shallow normalized copy of one ISLES26 datalist record.
    """
    normalized = dict(record)
    for key in (ISLES26_MODALITY_KEY, "label", "metadata_csv"):
        if key in normalized:
            normalized[key] = _resolve_path_value(basedir, normalized[key])
    return normalized


def _normalize_case_id(case_id: object) -> str:
    """
    Normalize case identity to a non-empty string.
    """
    if isinstance(case_id, list):
        if len(case_id) == 0:
            raise ValueError("ISLES26 record has empty list for 'caseID'.")
        case_id = case_id[0]
    if not _is_non_empty(case_id):
        raise ValueError("ISLES26 record requires non-empty 'caseID'.")
    return str(case_id)


def _normalize_t1_paths(t1_value: object, basedir: str) -> list[str]:
    """
    Normalize T1 field to a non-empty list of absolute/expanded paths.
    """
    if isinstance(t1_value, str):
        candidate_paths = [t1_value]
    elif isinstance(t1_value, list):
        candidate_paths = t1_value
    else:
        raise ValueError(
            "ISLES26 record field 'T1' must be either a string path or list of string paths."
        )

    resolved_paths: list[str] = []
    for path_value in candidate_paths:
        if not _is_non_empty(path_value):
            continue
        resolved = _resolve_path_value(basedir, path_value)
        resolved_paths.append(str(resolved))

    if len(resolved_paths) == 0:
        raise ValueError("ISLES26 record requires non-empty 'T1' path list.")
    return resolved_paths


def _normalize_case_record(record: Mapping[str, Any], basedir: str) -> Dict[str, Any]:
    """
    Validate and normalize one ISLES26 datalist record.
    """
    missing = [key for key in ISLES26_REQUIRED_RECORD_KEYS if key not in record]
    if missing:
        raise ValueError(
            "ISLES26 record is missing required keys: "
            f"{missing}. Required keys: {ISLES26_REQUIRED_RECORD_KEYS}."
        )

    normalized = _normalize_case_record_paths(record=record, basedir=basedir)
    normalized["caseID"] = _normalize_case_id(record.get("caseID"))

    try:
        normalized["fold"] = int(record.get("fold"))
    except Exception as exc:
        raise ValueError(
            f"ISLES26 record has invalid 'fold' value: {record.get('fold')}."
        ) from exc

    normalized[ISLES26_MODALITY_KEY] = _normalize_t1_paths(
        t1_value=record.get(ISLES26_MODALITY_KEY),
        basedir=basedir,
    )

    label_value = record.get("label")
    if not _is_non_empty(label_value):
        raise ValueError("ISLES26 record requires non-empty 'label' path.")
    normalized["label"] = str(_resolve_path_value(basedir, label_value))

    if "siteID" in record and _is_non_empty(record.get("siteID")):
        normalized["siteID"] = str(record.get("siteID"))

    if "metadata" in record:
        metadata = record.get("metadata")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError("ISLES26 record field 'metadata' must be a mapping when provided.")
        normalized["metadata"] = metadata

    if "metadata_csv" in record and _is_non_empty(record.get("metadata_csv")):
        normalized["metadata_csv"] = str(_resolve_path_value(basedir, record.get("metadata_csv")))

    return normalized


def datafold_read(datalist, basedir, fold=0, key="training"):
    """
    Read and normalize ISLES26 datalist entries with fold-based split.

    Behavior mirrors ISLES24 splitting semantics:
    - records with `record["fold"] != fold` go to train
    - records with `record["fold"] == fold` go to val
    """
    datalist_path = os.path.expanduser(str(datalist))
    basedir_path = os.path.expanduser(str(basedir))
    target_fold = int(fold)

    with open(datalist_path, encoding="utf-8") as handle:
        payload = json.load(handle)

    if key not in payload:
        raise ValueError(
            f"ISLES26 datalist missing key '{key}'. Available top-level keys: {list(payload.keys())}."
        )

    raw_records = payload[key]
    if not isinstance(raw_records, list):
        raise ValueError(
            f"ISLES26 datalist key '{key}' must contain a list, got: {type(raw_records).__name__}."
        )

    normalized_records: list[Dict[str, Any]] = []
    for index, record in enumerate(raw_records):
        if not isinstance(record, Mapping):
            raise ValueError(
                f"ISLES26 datalist entry at index {index} must be a mapping, "
                f"got: {type(record).__name__}."
            )
        normalized_records.append(
            _normalize_case_record(record=record, basedir=basedir_path)
        )

    train_records = [record for record in normalized_records if record.get("fold") != target_fold]
    val_records = [record for record in normalized_records if record.get("fold") == target_fold]
    return train_records, val_records


class ISLES26Dataset3D(torch.utils.data.Dataset):
    """
    ISLES26 3D full-volume dataset scaffold.

    Signature intentionally mirrors ISLES24Dataset3D for parity.
    """

    def __init__(
        self,
        directory,
        datalist_json,
        fold=0,
        transform=None,
        modalities=None,
        test_flag=False,
    ):
        super().__init__()
        self.directory = os.path.expanduser(str(directory))
        self.datalist_json = datalist_json
        self.fold = int(fold)
        self.transform = transform
        self.test_flag = bool(test_flag)
        self.modalities = _normalize_modalities(modalities)
        self.base_modalities = sorted(
            list(set(modality.split("_")[0] for modality in self.modalities))
        )

        train_files, val_files = datafold_read(
            datalist=self.datalist_json,
            basedir=self.directory,
            fold=self.fold,
        )
        self.database = val_files if self.test_flag else train_files

    def __len__(self):
        return len(self.database)

    def _process_modalities(self, data: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Process 3D modalities for ISLES26.

        Phase 4.3 keeps this intentionally simple (T1 passthrough).
        Phase 5 introduces richer T1 processing presets.
        """
        processed_images: Dict[str, torch.Tensor] = {}
        for modality_config in self.modalities:
            base_modality = modality_config.split("_")[0]
            if base_modality != ISLES26_MODALITY_KEY:
                raise ValueError(
                    "ISLES26Dataset3D currently supports only T1-derived modality tokens. "
                    f"Got modality '{modality_config}'."
                )
            if base_modality not in data:
                raise KeyError(
                    f"Missing modality volume '{base_modality}' for ISLES26 case."
                )
            # Keep raw intensity behavior for now; configurable T1 normalization arrives in Phase 5.
            processed_images[f"processed_{modality_config}"] = data[base_modality].float()
        return processed_images

    def __getitem__(self, index):
        filedict = self.database[index]
        data: Dict[str, torch.Tensor] = {}
        keys_to_load = self.base_modalities + ["label"]

        for key in keys_to_load:
            if key not in filedict or not filedict[key]:
                continue
            filepath = filedict[key][0] if isinstance(filedict[key], list) else filedict[key]
            if os.path.exists(filepath):
                nib_img = nibabel.load(filepath)
                data[key] = torch.tensor(nib_img.get_fdata(), dtype=torch.float32)

        processed_images = self._process_modalities(data)
        image_channels = [processed_images[f"processed_{modality}"] for modality in self.modalities]
        image = torch.stack(image_channels, dim=0)

        label = data.get("label")
        if label is None:
            label = torch.zeros_like(image_channels[0]).unsqueeze(0)
        else:
            label = label.unsqueeze(0)

        if self.transform:
            state = torch.get_rng_state()
            image = self.transform(image)
            torch.set_rng_state(state)
            label = self.transform(label)

        return image, label, filedict["caseID"]


class ISLES26Dataset2D(torch.utils.data.Dataset):
    """
    ISLES26 online 3D->2D slice dataset scaffold.

    Signature intentionally mirrors ISLES24Dataset2D for parity.
    """

    def __init__(
        self,
        directory,
        datalist_json,
        fold=0,
        transform=None,
        modalities=None,
        test_flag=False,
        image_size=32,
        use_caching=False,
        shared_cache=None,
        cache_lock=None,
        aug_cfg=None,
        is_training=False,
    ):
        super().__init__()
        self.directory = os.path.expanduser(str(directory))
        self.datalist_json = datalist_json
        self.fold = int(fold)
        self.transform = transform
        self.modalities = _normalize_modalities(modalities)
        self.base_modalities = sorted(
            list(set(modality.split("_")[0] for modality in self.modalities))
        )

        train_files, val_files = datafold_read(
            datalist=self.datalist_json,
            basedir=self.directory,
            fold=self.fold,
        )
        self.database = val_files if bool(test_flag) else train_files

        self.all_slices = []
        self.test_flag = bool(test_flag)
        self.image_size = int(image_size)
        self.use_caching = bool(use_caching)
        self._cache_prefix = "ts" if self.test_flag else "tr"
        self.shared_cache = shared_cache
        self._shared_cache_lock = cache_lock
        self.aug_cfg = aug_cfg
        self.is_training = bool(is_training)
        self.augmentation = None
        self.return_metadata = False

        if self.is_training and self.aug_cfg is not None:
            from src.data.augmentation import AugmentationPipeline2D

            self.augmentation = AugmentationPipeline2D(self.aug_cfg)
            print("Initialized augmentation pipeline for ISLES26 training dataset")

        for case_idx, filedict in enumerate(self.database):
            first_mod_key = self.base_modalities[0]
            filepath = (
                filedict[first_mod_key][0]
                if isinstance(filedict[first_mod_key], list)
                else filedict[first_mod_key]
            )
            if os.path.exists(filepath):
                num_slices = nibabel.load(filepath).shape[-1]
                self.all_slices.extend(
                    [(case_idx, slice_idx) for slice_idx in range(num_slices)]
                )

        self.cache = None
        self.cache_lock = None
        if self.use_caching:
            if self.shared_cache is not None:
                self.cache = self.shared_cache
                self.cache_lock = self._shared_cache_lock or threading.Lock()
            else:
                self.cache = {}
                self.cache_lock = threading.Lock()

    def __len__(self):
        return len(self.all_slices)

    def _process_modalities(
        self, data_slice: Mapping[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Process 2D modalities for ISLES26.

        Phase 4.4 keeps this intentionally simple (T1 passthrough).
        Phase 5 introduces richer T1 processing presets.
        """
        processed_images: Dict[str, torch.Tensor] = {}
        for modality_config in self.modalities:
            base_modality = modality_config.split("_")[0]
            if base_modality != ISLES26_MODALITY_KEY:
                raise ValueError(
                    "ISLES26Dataset2D currently supports only T1-derived modality tokens. "
                    f"Got modality '{modality_config}'."
                )
            raw_data = data_slice.get(base_modality)
            if raw_data is None:
                raise KeyError(
                    f"Missing modality slice '{base_modality}' for ISLES26 case."
                )
            processed_images[f"processed_{modality_config}"] = raw_data.float()
        return processed_images

    def _build_slice_metadata(self, filedict: Mapping[str, Any], slice_idx: int) -> Dict[str, Any]:
        metadata = {
            "caseID": str(filedict.get("caseID", "")),
            "slice_index": int(slice_idx),
            "source_path": (
                filedict.get(ISLES26_MODALITY_KEY, [""])[0]
                if isinstance(filedict.get(ISLES26_MODALITY_KEY, []), list)
                else filedict.get(ISLES26_MODALITY_KEY, "")
            ),
            "virtual_path": ISLES26_VIRTUAL_PATH_TEMPLATE.format(
                case_id=str(filedict.get("caseID", "")),
                slice_idx=int(slice_idx),
            ),
        }
        if "siteID" in filedict:
            metadata["siteID"] = filedict["siteID"]
        if "metadata_csv" in filedict:
            metadata["metadata_csv"] = filedict["metadata_csv"]
        if "metadata" in filedict:
            metadata["metadata"] = filedict["metadata"]
        return metadata

    def __getitem__(self, index):
        case_idx, slice_idx = self.all_slices[index]
        filedict = self.database[case_idx]
        cache_key = (self._cache_prefix, int(case_idx))

        if self.use_caching:
            if cache_key not in self.cache:
                with self.cache_lock:
                    if cache_key not in self.cache:
                        data = {}
                        keys_to_load = self.base_modalities + ["label"]
                        for key in keys_to_load:
                            if key not in filedict or not filedict[key]:
                                continue
                            filepath = (
                                filedict[key][0]
                                if isinstance(filedict[key], list)
                                else filedict[key]
                            )
                            if os.path.exists(filepath):
                                nib_img = nibabel.load(filepath)
                                data[key] = torch.from_numpy(
                                    nib_img.get_fdata().astype(np.float32)
                                )
                        self.cache[cache_key] = data
            data_slice: Dict[str, torch.Tensor] = {}
            for key in self.cache[cache_key]:
                data_slice[key] = self.cache[cache_key][key][..., slice_idx]
        else:
            data_slice = {}
            keys_to_load = self.base_modalities + ["label"]
            for key in keys_to_load:
                if key not in filedict or not filedict[key]:
                    continue
                filepath = (
                    filedict[key][0]
                    if isinstance(filedict[key], list)
                    else filedict[key]
                )
                if os.path.exists(filepath):
                    nib_img = nibabel.load(filepath)
                    volume_data = torch.from_numpy(
                        nib_img.get_fdata().astype(np.float32)
                    )
                    data_slice[key] = volume_data[..., slice_idx]

        processed_images = self._process_modalities(data_slice)
        image_channels = [
            processed_images[f"processed_{modality}"] for modality in self.modalities
        ]
        image = torch.stack(image_channels, dim=0)

        label = data_slice.get("label")
        if label is None:
            label = torch.zeros_like(image_channels[0]).unsqueeze(0)
        else:
            label = label.unsqueeze(0)

        resizer = Resize(spatial_size=(self.image_size, self.image_size))
        image = resizer(image)
        label = resizer(label)

        if self.augmentation is not None:
            data_dict = {"image": image, "label": label}
            data_dict = self.augmentation(data_dict)
            image = data_dict["image"]
            label = data_dict["label"]

        if self.transform:
            state = torch.get_rng_state()
            image = self.transform(image)
            torch.set_rng_state(state)
            label = self.transform(label)

        virtual_path = ISLES26_VIRTUAL_PATH_TEMPLATE.format(
            case_id=filedict["caseID"],
            slice_idx=int(slice_idx),
        )
        if self.return_metadata:
            return image, label, virtual_path, self._build_slice_metadata(filedict, slice_idx)
        return image, label, virtual_path


ISLES26OnlineProc2D = ISLES26Dataset2D


def phase_marker() -> str:
    """
    Return a stable marker string for incremental refactor checks.
    """
    return "loader_stack.isles26_loader.phase4_4"


__all__ = [
    "ISLES26_MODALITY_KEY",
    "ISLES26_SUPPORTED_MODALITY_KEYS",
    "ISLES26_REQUIRED_RECORD_KEYS",
    "ISLES26_OPTIONAL_RECORD_KEYS",
    "ISLES26_VIRTUAL_PATH_TEMPLATE",
    "_normalize_modalities",
    "_normalize_case_record_paths",
    "datafold_read",
    "ISLES26Dataset3D",
    "ISLES26Dataset2D",
    "ISLES26OnlineProc2D",
]
