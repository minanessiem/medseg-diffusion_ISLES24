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
import torch
from monai.transforms import (
    Compose,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    Resize,
    SpatialPadd,
    Spacingd,
)

from src.data.augmentation import AugmentationPipeline2D, AugmentationPipeline3D
from src.utils.loader_monai_utils import build_monai_compose_safe
from src.utils.loader_transforms import (
    MergeProcessedChannelsTransform,
    ProcessModalitiesTransform,
)
from src.utils.loader_utils import LoaderDataUtils

ISLES26_MODALITY_KEY = "T1"
ISLES26_SUPPORTED_MODALITY_KEYS = (ISLES26_MODALITY_KEY,)
ISLES26_SUPPORTED_PREPROCESSING_KEYS = ("RAW",)
ISLES26_REQUIRED_RECORD_KEYS = ("fold", "caseID", ISLES26_MODALITY_KEY, "label")
ISLES26_OPTIONAL_RECORD_KEYS = ("siteID", "metadata_csv", "metadata")
ISLES26_VIRTUAL_PATH_TEMPLATE = "{case_id}_slice{slice_idx}"
ISLES26_PATCH_PATH_TEMPLATE = "{case_id}_patch{patch_idx}"


def _parse_modality_config(modality_config: str) -> Tuple[str, str]:
    try:
        return LoaderDataUtils.parse_modality_token(modality_config)
    except ValueError as exc:
        raise ValueError(
            "ISLES26 modality config must use '<modality_key>_<preprocessing_key>' format. "
            f"Got '{modality_config}'."
        ) from exc


def _resolve_base_modality(modality_config: str) -> str:
    base_modality, _ = _parse_modality_config(modality_config)
    return base_modality


def _resolve_t1_preprocessing_params(
    preprocessing_key: str,
    data_stats: Mapping[str, float],
) -> Dict[str, Any]:
    if preprocessing_key == "RAW":
        return {}

    raise NotImplementedError(
        "Unsupported ISLES26 T1 preprocessing key "
        f"'{preprocessing_key}'. Supported keys: {ISLES26_SUPPORTED_PREPROCESSING_KEYS}. "
        f"Observed stats: {dict(data_stats)}"
    )


def _process_t1_raw(raw_tensor: torch.Tensor, **_: Any) -> torch.Tensor:
    return raw_tensor.float()


def _process_t1_modality(
    modality_config: str,
    raw_tensor: torch.Tensor,
) -> torch.Tensor:
    base_modality, preprocessing_key = _parse_modality_config(modality_config)
    if base_modality != ISLES26_MODALITY_KEY:
        raise ValueError(
            "ISLES26 currently supports only T1-based modality tokens. "
            f"Got '{modality_config}'."
        )

    data_stats = LoaderDataUtils.compute_tensor_data_stats(raw_tensor)
    params = _resolve_t1_preprocessing_params(
        preprocessing_key=preprocessing_key,
        data_stats=data_stats,
    )
    processor = _process_t1_raw if preprocessing_key == "RAW" else None
    if processor is None:
        raise NotImplementedError(
            "Unsupported ISLES26 T1 preprocessing key "
            f"'{preprocessing_key}'. Supported keys: {ISLES26_SUPPORTED_PREPROCESSING_KEYS}."
        )
    return processor(raw_tensor, **params)


def _build_common_preprocess_transforms(
    modalities: Sequence[str],
    preprocessing_configs: Mapping[str, Any],
) -> list[Any]:
    base_modalities = sorted(set(_resolve_base_modality(modality) for modality in modalities))
    keys_to_load = [*base_modalities, "label"]

    common_cfg = preprocessing_configs["common"]
    orientation_cfg = common_cfg["orientation"]
    spacing_cfg = common_cfg["spacing"]

    transforms: list[Any] = [
        LoadImaged(
            keys=keys_to_load,
            reader="NibabelReader",
            ensure_channel_first=True,
        )
    ]
    if bool(orientation_cfg["enabled"]):
        transforms.append(
            Orientationd(
                keys=keys_to_load,
                axcodes=str(orientation_cfg["axcodes"]),
            )
        )

    if bool(spacing_cfg["enabled"]):
        pixdim = LoaderDataUtils.as_float_tuple(
            spacing_cfg["pixdim"],
            expected_len=3,
            field_name="dataset.preprocessing_configs.common.spacing.pixdim",
        )
        image_interp = str(spacing_cfg["interpolation"]["image"])
        label_interp = str(spacing_cfg["interpolation"]["label"])
        spacing_modes = tuple([image_interp] * len(base_modalities) + [label_interp])
        transforms.append(
            Spacingd(
                keys=keys_to_load,
                pixdim=pixdim,
                mode=spacing_modes,
            )
        )

    transforms.extend(
        [
            ProcessModalitiesTransform(
                modalities,
                resolve_base_modality=_resolve_base_modality,
                process_modality=_process_t1_modality,
            ),
            MergeProcessedChannelsTransform(
                modalities,
                resolve_base_modality=_resolve_base_modality,
            ),
            EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        ]
    )
    return transforms


def build_common_preprocessed_volume_pipeline(
    modalities: Sequence[str],
    preprocessing_configs: Mapping[str, Any],
) -> Compose:
    return build_monai_compose_safe(
        _build_common_preprocess_transforms(modalities, preprocessing_configs)
    )


def build_online_slices_3d_to_2d_case_pipeline(
    modalities: Sequence[str],
    preprocessing_configs: Mapping[str, Any],
) -> Compose:
    return build_common_preprocessed_volume_pipeline(
        modalities=modalities,
        preprocessing_configs=preprocessing_configs,
    )


def build_full_volumes_3d_pipeline(
    modalities: Sequence[str],
    preprocessing_configs: Mapping[str, Any],
) -> Compose:
    transforms = _build_common_preprocess_transforms(modalities, preprocessing_configs)
    fullvol_cfg = preprocessing_configs["full_volumes_3d"]
    if bool(fullvol_cfg["pad_to_divisible"]):
        roi_3d = LoaderDataUtils.as_int_tuple(
            preprocessing_configs["roi"]["volume_3d"],
            expected_len=3,
            field_name="dataset.preprocessing_configs.roi.volume_3d",
        )
        transforms.append(SpatialPadd(keys=["image", "label"], spatial_size=roi_3d))
    return build_monai_compose_safe(transforms)


def _build_random_patches_3d_sampling_transforms(
    preprocessing_configs: Mapping[str, Any],
    patches_per_volume: Optional[int] = None,
) -> list[Any]:
    transforms: list[Any] = []
    random_patch_cfg = preprocessing_configs["random_patches_3d"]
    roi_3d = LoaderDataUtils.as_int_tuple(
        preprocessing_configs["roi"]["volume_3d"],
        expected_len=3,
        field_name="dataset.preprocessing_configs.roi.volume_3d",
    )
    if patches_per_volume is None:
        num_samples = int(random_patch_cfg["patches_per_volume"]["train"])
    else:
        num_samples = int(patches_per_volume)
    if num_samples <= 0:
        raise ValueError(
            "random_patches_3d requires patches_per_volume to be > 0, "
            f"got {num_samples}."
        )

    crop_cfg = random_patch_cfg["rand_crop_by_pos_neg_label"]
    transforms.append(
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=roi_3d,
            pos=float(crop_cfg["pos"]),
            neg=float(crop_cfg["neg"]),
            num_samples=num_samples,
            image_key="image",
            image_threshold=float(crop_cfg["image_threshold"]),
            allow_smaller=bool(crop_cfg["allow_smaller"]),
        )
    )
    if bool(random_patch_cfg["pad_to_divisible"]):
        transforms.append(SpatialPadd(keys=["image", "label"], spatial_size=roi_3d))
    transforms.append(EnsureTyped(keys=["image", "label"], dtype=torch.float32))
    return transforms


def build_random_patches_3d_pipeline(
    modalities: Sequence[str],
    preprocessing_configs: Mapping[str, Any],
    patches_per_volume: Optional[int] = None,
    include_common_preprocess: bool = True,
) -> Compose:
    transforms: list[Any] = []
    if include_common_preprocess:
        transforms.extend(_build_common_preprocess_transforms(modalities, preprocessing_configs))
    transforms.extend(
        _build_random_patches_3d_sampling_transforms(
            preprocessing_configs=preprocessing_configs,
            patches_per_volume=patches_per_volume,
        )
    )
    return build_monai_compose_safe(transforms)


def _normalize_modalities(modalities: Optional[Sequence[str]]) -> Tuple[str, ...]:
    """
    Normalize modality contract for ISLES26.

    Modality tokens must follow `<modality_key>_<preprocessing_key>` format
    (for example `T1_RAW`).
    Defaults to the single-modality `T1_RAW` channel when no explicit list is given.
    """
    if modalities is None:
        return (f"{ISLES26_MODALITY_KEY}_RAW",)
    normalized = tuple(str(modality) for modality in modalities)
    if len(normalized) == 0:
        raise ValueError("ISLES26 loaders require a non-empty modalities list.")
    for modality in normalized:
        base_modality, preprocessing_key = _parse_modality_config(modality)
        if base_modality != ISLES26_MODALITY_KEY:
            raise ValueError(
                "ISLES26 loaders currently support only T1-derived modality tokens. "
                f"Got modality '{modality}'."
            )
        if preprocessing_key not in ISLES26_SUPPORTED_PREPROCESSING_KEYS:
            raise ValueError(
                "Unsupported ISLES26 preprocessing key "
                f"'{preprocessing_key}' in modality '{modality}'. "
                f"Supported keys: {ISLES26_SUPPORTED_PREPROCESSING_KEYS}."
            )
    return normalized


def _normalize_case_record_paths(record: Mapping[str, Any], basedir: str) -> Dict[str, Any]:
    """
    Return a shallow normalized copy of one ISLES26 datalist record.
    """
    normalized = dict(record)
    for key in (ISLES26_MODALITY_KEY, "label", "metadata_csv"):
        if key in normalized:
            normalized[key] = LoaderDataUtils.resolve_path_value(basedir, normalized[key])
    return normalized


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
        if not LoaderDataUtils.is_non_empty(path_value):
            continue
        resolved = LoaderDataUtils.resolve_path_value(basedir, path_value)
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
    normalized["caseID"] = LoaderDataUtils.normalize_case_id(
        record.get("caseID"),
        field_name="caseID",
    )

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
    if not LoaderDataUtils.is_non_empty(label_value):
        raise ValueError("ISLES26 record requires non-empty 'label' path.")
    normalized["label"] = str(LoaderDataUtils.resolve_path_value(basedir, label_value))

    if "siteID" in record and LoaderDataUtils.is_non_empty(record.get("siteID")):
        normalized["siteID"] = str(record.get("siteID"))

    if "metadata" in record:
        metadata = record.get("metadata")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError("ISLES26 record field 'metadata' must be a mapping when provided.")
        normalized["metadata"] = metadata

    if "metadata_csv" in record and LoaderDataUtils.is_non_empty(
        record.get("metadata_csv")
    ):
        normalized["metadata_csv"] = str(
            LoaderDataUtils.resolve_path_value(basedir, record.get("metadata_csv"))
        )

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
        image_size=32,
        preprocessing_configs: Optional[Mapping[str, Any]] = None,
        aug_cfg=None,
        is_training=False,
    ):
        super().__init__()
        self.directory = os.path.expanduser(str(directory))
        self.datalist_json = datalist_json
        self.fold = int(fold)
        self.transform = transform
        self.test_flag = bool(test_flag)
        self.image_size = int(image_size)
        self.aug_cfg = aug_cfg
        self.is_training = bool(is_training)
        if not isinstance(preprocessing_configs, Mapping):
            raise ValueError(
                "ISLES26Dataset3D requires dataset.preprocessing_configs mapping. "
                f"Got: {type(preprocessing_configs).__name__}."
            )
        self.preprocessing_configs = dict(preprocessing_configs)
        self.modalities = _normalize_modalities(modalities)
        self.base_modalities = sorted(
            list(set(_parse_modality_config(modality)[0] for modality in self.modalities))
        )

        train_files, val_files = datafold_read(
            datalist=self.datalist_json,
            basedir=self.directory,
            fold=self.fold,
        )
        self.database = val_files if self.test_flag else train_files
        self.full_volume_pipeline = build_full_volumes_3d_pipeline(
            modalities=self.modalities,
            preprocessing_configs=self.preprocessing_configs,
        )
        self.augmentation = None
        if self.is_training and self.aug_cfg is not None:
            self.augmentation = AugmentationPipeline3D(self.aug_cfg)
            print("Initialized 3D augmentation pipeline for ISLES26 full-volume training dataset")

    def __len__(self):
        return len(self.database)

    def _load_case_volume(self, filedict: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        case_input = LoaderDataUtils.resolve_case_input_paths(
            filedict=filedict,
            base_modalities=self.base_modalities,
        )
        case_data = self.full_volume_pipeline(case_input)
        return {
            "image": case_data["image"],
            "label": case_data["label"],
        }

    def __getitem__(self, index):
        filedict = self.database[index]
        case_data = self._load_case_volume(filedict)
        image = case_data["image"]
        label = case_data["label"]

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

        return image, label, filedict["caseID"]


class ISLES26RandomPatches3D(torch.utils.data.Dataset):
    """
    ISLES26 random 3D patch dataset.

    Flatten policy:
    - __len__ = num_cases * patches_per_volume.train
    - __getitem__ returns one patch sample
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
        preprocessing_configs: Optional[Mapping[str, Any]] = None,
        aug_cfg=None,
        is_training=False,
    ):
        super().__init__()
        self.directory = os.path.expanduser(str(directory))
        self.datalist_json = datalist_json
        self.fold = int(fold)
        self.transform = transform
        self.test_flag = bool(test_flag)
        self.image_size = int(image_size)
        self.aug_cfg = aug_cfg
        self.is_training = bool(is_training)
        if not isinstance(preprocessing_configs, Mapping):
            raise ValueError(
                "ISLES26RandomPatches3D requires dataset.preprocessing_configs mapping. "
                f"Got: {type(preprocessing_configs).__name__}."
            )
        self.preprocessing_configs = dict(preprocessing_configs)
        self.modalities = _normalize_modalities(modalities)
        self.base_modalities = sorted(
            list(set(_parse_modality_config(modality)[0] for modality in self.modalities))
        )
        train_files, val_files = datafold_read(
            datalist=self.datalist_json,
            basedir=self.directory,
            fold=self.fold,
        )
        self.database = val_files if self.test_flag else train_files

        self.patches_per_volume = int(
            self.preprocessing_configs["random_patches_3d"]["patches_per_volume"]["train"]
        )
        if self.patches_per_volume <= 0:
            raise ValueError(
                "dataset.preprocessing_configs.random_patches_3d.patches_per_volume.train "
                f"must be > 0, got {self.patches_per_volume}."
            )
        self.preprocessed_volume_pipeline = build_common_preprocessed_volume_pipeline(
            modalities=self.modalities,
            preprocessing_configs=self.preprocessing_configs,
        )
        self.random_patch_sampler_pipeline = build_random_patches_3d_pipeline(
            modalities=self.modalities,
            preprocessing_configs=self.preprocessing_configs,
            patches_per_volume=self.patches_per_volume,
            include_common_preprocess=False,
        )
        self.augmentation = None
        if self.is_training and self.aug_cfg is not None:
            self.augmentation = AugmentationPipeline3D(self.aug_cfg)
            print("Initialized 3D augmentation pipeline for ISLES26 random-patch training dataset")

    def __len__(self):
        return len(self.database) * self.patches_per_volume

    def _load_case_patches(self, filedict: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        case_input = LoaderDataUtils.resolve_case_input_paths(
            filedict=filedict,
            base_modalities=self.base_modalities,
        )
        case_data = self.preprocessed_volume_pipeline(case_input)
        data_dict = {
            "image": case_data["image"],
            "label": case_data["label"],
        }
        patch_samples = LoaderDataUtils.as_sample_list(
            self.random_patch_sampler_pipeline(data_dict)
        )
        if len(patch_samples) != self.patches_per_volume:
            raise RuntimeError(
                "Random patch pipeline returned unexpected number of samples. "
                f"Expected {self.patches_per_volume}, got {len(patch_samples)}."
            )
        return patch_samples

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Patch index out of range: {index}. Dataset length is {len(self)}."
            )
        case_idx = int(index) // self.patches_per_volume
        patch_idx = int(index) % self.patches_per_volume
        filedict = self.database[case_idx]

        patch_samples = self._load_case_patches(filedict)
        patch_sample = patch_samples[patch_idx]
        image = patch_sample["image"]
        label = patch_sample["label"]

        # Random-patch mode policy: crop first, then augment per returned patch.
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

        patch_path = ISLES26_PATCH_PATH_TEMPLATE.format(
            case_id=filedict["caseID"],
            patch_idx=int(patch_idx),
        )
        return image, label, patch_path


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
        preprocessing_configs: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        self.directory = os.path.expanduser(str(directory))
        self.datalist_json = datalist_json
        self.fold = int(fold)
        self.transform = transform
        self.modalities = _normalize_modalities(modalities)
        self.base_modalities = sorted(
            list(set(_parse_modality_config(modality)[0] for modality in self.modalities))
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
        if not isinstance(preprocessing_configs, Mapping):
            raise ValueError(
                "ISLES26Dataset2D requires dataset.preprocessing_configs mapping. "
                f"Got: {type(preprocessing_configs).__name__}."
            )
        self.preprocessing_configs = dict(preprocessing_configs)
        self.slice_axis = int(
            self.preprocessing_configs["online_slices_3d_to_2d"]["slice_axis"]
        )
        if self.slice_axis not in (0, 1, 2):
            raise ValueError(
                "dataset.preprocessing_configs.online_slices_3d_to_2d.slice_axis "
                f"must be one of {{0,1,2}}, got {self.slice_axis}."
            )
        self.slice_order = str(
            self.preprocessing_configs["online_slices_3d_to_2d"]["slice_order"]
        )
        if self.slice_order not in {"sequential", "reverse"}:
            raise ValueError(
                "dataset.preprocessing_configs.online_slices_3d_to_2d.slice_order "
                f"must be one of {{sequential, reverse}}, got '{self.slice_order}'."
            )
        self.slice_roi = LoaderDataUtils.as_int_tuple(
            self.preprocessing_configs["roi"]["slice_2d"],
            expected_len=2,
            field_name="dataset.preprocessing_configs.roi.slice_2d",
        )
        self.case_pipeline = build_online_slices_3d_to_2d_case_pipeline(
            modalities=self.modalities,
            preprocessing_configs=self.preprocessing_configs,
        )
        self.image_slice_resize = Resize(
            spatial_size=self.slice_roi,
            mode="bilinear",
        )
        self.label_slice_resize = Resize(
            spatial_size=self.slice_roi,
            mode="nearest",
        )
        self.use_caching = bool(use_caching)
        self._cache_prefix = "ts" if self.test_flag else "tr"
        self.shared_cache = shared_cache
        self._shared_cache_lock = cache_lock
        self.aug_cfg = aug_cfg
        self.is_training = bool(is_training)
        self.augmentation = None
        self.return_metadata = False

        if self.is_training and self.aug_cfg is not None:
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
                num_slices = int(nibabel.load(filepath).shape[self.slice_axis])
                if self.slice_order == "reverse":
                    slice_indices = range(num_slices - 1, -1, -1)
                else:
                    slice_indices = range(num_slices)
                self.all_slices.extend(
                    [(case_idx, slice_idx) for slice_idx in slice_indices]
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

    def _load_case_volume(self, filedict: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        case_input = LoaderDataUtils.resolve_case_input_paths(
            filedict=filedict,
            base_modalities=self.base_modalities,
        )
        case_data = self.case_pipeline(case_input)
        return {
            "image": case_data["image"],
            "label": case_data["label"],
        }

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
                        self.cache[cache_key] = self._load_case_volume(filedict)
            case_data = self.cache[cache_key]
        else:
            case_data = self._load_case_volume(filedict)

        image_volume = case_data["image"]
        label_volume = case_data["label"]

        # image/label are [C, X, Y, Z] after common preprocessing.
        slice_dim = int(self.slice_axis) + 1
        image = image_volume.select(dim=slice_dim, index=int(slice_idx))
        label = label_volume.select(dim=slice_dim, index=int(slice_idx))

        image = self.image_slice_resize(image)
        label = self.label_slice_resize(label)

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
    return "loader_stack.isles26_loader.phase4_7"


__all__ = [
    "ISLES26_MODALITY_KEY",
    "ISLES26_SUPPORTED_MODALITY_KEYS",
    "ISLES26_SUPPORTED_PREPROCESSING_KEYS",
    "ISLES26_REQUIRED_RECORD_KEYS",
    "ISLES26_OPTIONAL_RECORD_KEYS",
    "ISLES26_VIRTUAL_PATH_TEMPLATE",
    "ISLES26_PATCH_PATH_TEMPLATE",
    "build_common_preprocessed_volume_pipeline",
    "build_online_slices_3d_to_2d_case_pipeline",
    "build_full_volumes_3d_pipeline",
    "build_random_patches_3d_pipeline",
    "_normalize_modalities",
    "_normalize_case_record_paths",
    "datafold_read",
    "ISLES26Dataset3D",
    "ISLES26RandomPatches3D",
    "ISLES26Dataset2D",
    "ISLES26OnlineProc2D",
]
