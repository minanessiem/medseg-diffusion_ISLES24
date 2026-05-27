import torch
from torch.utils.data import DataLoader, Subset, default_collate
from torch.utils.data.distributed import DistributedSampler

from omegaconf import OmegaConf
import threading
# Compatibility facade: shared internals are migrated into loader_stack
# while keeping public imports stable at src.data.loaders.
from src.data.loader_stack.core import _build_loader_kwargs, _is_set
from src.data.loader_stack.factory import resolve_loader_contract
from src.data.loader_stack.registry import DatasetCapabilities
from src.utils.distribution_utils import resolve_strategy, resolve_train_batch_sizes

from src.data.loader_stack.isles24_loader import (
    ISLES24Dataset2D,
    ISLES24Dataset3D,
    ISLES24RandomPatches3D,
    ISLES24NNUNet2D,
    ISLES24OnlineProc2D,
    datafold_read,
)
from src.data.loader_stack.isles26_loader import (
    ISLES26Dataset2D,
    ISLES26Dataset3D,
    ISLES26OnlineProc2D,
    ISLES26RandomPatches3D,
)


_ISLES24_LOADER_MODULE = "src.data.loader_stack.isles24_loader"
_ISLES26_LOADER_MODULE = "src.data.loader_stack.isles26_loader"
_LOADER_SMOKE_ITEMS = 10
try:
    from monai.data.meta_tensor import MetaTensor

    _META_TENSOR_TYPES = (MetaTensor,)
except Exception:  # pragma: no cover - MONAI should be present in runtime envs
    _META_TENSOR_TYPES = ()
_ACTIVE_RUNTIME_ROUTES = {
    (_ISLES24_LOADER_MODULE, "online_slices_3d_to_2d"): {"legacy-runtime"},
    (_ISLES24_LOADER_MODULE, "nnunet_slices_2d"): {"legacy-runtime"},
    (_ISLES24_LOADER_MODULE, "full_volumes_3d"): {"legacy-runtime"},
    (_ISLES24_LOADER_MODULE, "random_patches_3d"): {"legacy-runtime"},
    (_ISLES26_LOADER_MODULE, "online_slices_3d_to_2d"): {"online-runtime"},
    (_ISLES26_LOADER_MODULE, "nnunet_slices_2d"): {"online-runtime"},
    (_ISLES26_LOADER_MODULE, "full_volumes_3d"): {"online-runtime"},
    (_ISLES26_LOADER_MODULE, "random_patches_3d"): {"online-runtime"},
}


def _strip_meta_tensors(sample):
    """
    Recursively convert MONAI MetaTensor leaves to plain torch.Tensor.

    This avoids torch default_collate shared-storage resize issues seen in
    multi-worker validation with val_batch_size > 1.
    """
    if _META_TENSOR_TYPES and isinstance(sample, _META_TENSOR_TYPES):
        return sample.as_tensor()
    if isinstance(sample, dict):
        return {k: _strip_meta_tensors(v) for k, v in sample.items()}
    if isinstance(sample, tuple):
        return tuple(_strip_meta_tensors(v) for v in sample)
    if isinstance(sample, list):
        return [_strip_meta_tensors(v) for v in sample]
    return sample


def _meta_safe_collate(batch):
    normalized_batch = [_strip_meta_tensors(sample) for sample in batch]
    return default_collate(normalized_batch)


def _resolve_active_dataset_contract(cfg) -> tuple[str, str, DatasetCapabilities]:
    """
    Resolve dataset/mode through the registry-backed factory.

    Runtime remains scoped to dataset loaders marked as active.
    """
    resolution = resolve_loader_contract(
        dataset_id=OmegaConf.select(cfg, "dataset.id"),
        dataset_name=OmegaConf.select(cfg, "dataset.name"),
        loader_mode=OmegaConf.select(cfg, "data_mode.loader_mode"),
    )
    return resolution.dataset_id, resolution.loader_mode, resolution.capabilities


def validate_dataset_contract(cfg):
    """Validate explicit data contract before constructing datasets/dataloaders."""
    _dataset_id, loader_mode, _capabilities = _resolve_active_dataset_contract(cfg)
    dim = OmegaConf.select(cfg, "data_mode.dim")
    per_side_context_slices = OmegaConf.select(cfg, "data_mode.per_side_context_slices")
    channel_layout = OmegaConf.select(cfg, "data_mode.channel_layout")
    modalities = OmegaConf.select(cfg, "dataset.modalities")
    num_modalities = OmegaConf.select(cfg, "dataset.num_modalities")

    # Accept plain Python sequences and Hydra ListConfig for modality contracts.
    modalities_is_sequence = isinstance(modalities, (list, tuple)) or OmegaConf.is_list(modalities)
    if not modalities_is_sequence or len(modalities) == 0:
        raise ValueError("dataset.modalities must be a non-empty list.")
    if int(num_modalities) != len(modalities):
        raise ValueError(
            "Global invariant violated: len(dataset.modalities) must equal dataset.num_modalities. "
            f"Got len(modalities)={len(modalities)}, num_modalities={num_modalities}."
        )

    runtime_required = [
        "data_runtime.train_batch_size",
        "data_runtime.test_batch_size",
        "data_runtime.num_train_workers",
        "data_runtime.num_valid_workers",
        "data_runtime.num_test_workers",
        "data_runtime.use_caching",
        "data_runtime.use_shared_cache",
        "data_runtime.train_prefetch_factor",
        "data_runtime.test_prefetch_factor",
        "data_runtime.loader_smoke_testing",
    ]
    for key in runtime_required:
        value = OmegaConf.select(cfg, key)
        if value is None:
            raise ValueError(f"Missing required runtime key: {key}")

    if loader_mode == "online_slices_3d_to_2d":
        if dim != "2d":
            raise ValueError("online_slices_3d_to_2d requires data_mode.dim='2d'.")
        if not _is_set(OmegaConf.select(cfg, "data_io.paths.data_root")):
            raise ValueError("online_slices_3d_to_2d requires data_io.paths.data_root.")
        if not _is_set(OmegaConf.select(cfg, "data_io.paths.split_file")):
            raise ValueError("online_slices_3d_to_2d requires data_io.paths.split_file.")
        if per_side_context_slices is not None and int(per_side_context_slices) != 0:
            raise ValueError(
                "data_mode.per_side_context_slices is currently supported only for "
                "loader_mode='nnunet_slices_2d'."
            )
        if _is_set(channel_layout):
            raise ValueError(
                "data_mode.channel_layout is currently supported only for "
                "loader_mode='nnunet_slices_2d'."
            )
        if OmegaConf.select(cfg, "dataset.preprocessing_configs") is None:
            raise ValueError(
                "online_slices_3d_to_2d requires dataset.preprocessing_configs."
            )

    elif loader_mode == "nnunet_slices_2d":
        if dim != "2d":
            raise ValueError("nnunet_slices_2d requires data_mode.dim='2d'.")
        if not _is_set(OmegaConf.select(cfg, "data_io.paths.nnunet_root")):
            raise ValueError("nnunet_slices_2d requires data_io.paths.nnunet_root.")
        if not _is_set(OmegaConf.select(cfg, "dataset.nnunet.dataset_id")):
            raise ValueError("nnunet_slices_2d requires dataset.nnunet.dataset_id.")
        if not _is_set(OmegaConf.select(cfg, "dataset.nnunet.dataset_name")):
            raise ValueError("nnunet_slices_2d requires dataset.nnunet.dataset_name.")
        if per_side_context_slices is None:
            raise ValueError("nnunet_slices_2d requires data_mode.per_side_context_slices.")
        if int(per_side_context_slices) < 0:
            raise ValueError(
                "nnunet_slices_2d requires data_mode.per_side_context_slices >= 0."
            )
        if not _is_set(channel_layout):
            raise ValueError("nnunet_slices_2d requires data_mode.channel_layout.")
        if str(channel_layout) not in {"slice_major", "modality_major"}:
            raise ValueError(
                "nnunet_slices_2d requires data_mode.channel_layout in "
                "{slice_major, modality_major}."
            )

    elif loader_mode == "full_volumes_3d":
        if dim != "3d":
            raise ValueError("full_volumes_3d requires data_mode.dim='3d'.")
        if not _is_set(OmegaConf.select(cfg, "data_io.paths.data_root")):
            raise ValueError("full_volumes_3d requires data_io.paths.data_root.")
        if not _is_set(OmegaConf.select(cfg, "data_io.paths.split_file")):
            raise ValueError("full_volumes_3d requires data_io.paths.split_file.")
        if per_side_context_slices is not None and int(per_side_context_slices) != 0:
            raise ValueError(
                "data_mode.per_side_context_slices is currently supported only for "
                "loader_mode='nnunet_slices_2d'."
            )
        if _is_set(channel_layout):
            raise ValueError(
                "data_mode.channel_layout is currently supported only for "
                "loader_mode='nnunet_slices_2d'."
            )
        if OmegaConf.select(cfg, "dataset.preprocessing_configs") is None:
            raise ValueError("full_volumes_3d requires dataset.preprocessing_configs.")
    elif loader_mode == "random_patches_3d":
        if dim != "3d":
            raise ValueError("random_patches_3d requires data_mode.dim='3d'.")
        if not _is_set(OmegaConf.select(cfg, "data_io.paths.data_root")):
            raise ValueError("random_patches_3d requires data_io.paths.data_root.")
        if not _is_set(OmegaConf.select(cfg, "data_io.paths.split_file")):
            raise ValueError("random_patches_3d requires data_io.paths.split_file.")
        if per_side_context_slices is not None and int(per_side_context_slices) != 0:
            raise ValueError(
                "data_mode.per_side_context_slices is currently supported only for "
                "loader_mode='nnunet_slices_2d'."
            )
        if _is_set(channel_layout):
            raise ValueError(
                "data_mode.channel_layout is currently supported only for "
                "loader_mode='nnunet_slices_2d'."
            )
        if OmegaConf.select(cfg, "dataset.preprocessing_configs") is None:
            raise ValueError("random_patches_3d requires dataset.preprocessing_configs.")


def get_dataloaders(cfg):
    validate_dataset_contract(cfg)
    dataset_id, loader_mode, capabilities = _resolve_active_dataset_contract(cfg)
    route_key = (capabilities.loader_module, loader_mode)
    allowed_states = _ACTIVE_RUNTIME_ROUTES.get(route_key)
    if allowed_states is None:
        raise NotImplementedError(
            "Resolved dataset route is not wired in src.data.loaders runtime dispatch. "
            f"dataset={dataset_id}, loader_mode={loader_mode}, "
            f"loader_module={capabilities.loader_module}."
        )
    if capabilities.implementation_state not in allowed_states:
        raise NotImplementedError(
            "Resolved dataset route is registered but not active for current implementation_state. "
            f"dataset={dataset_id}, loader_mode={loader_mode}, "
            f"loader_module={capabilities.loader_module}, "
            f"implementation_state={capabilities.implementation_state}."
        )

    aug_cfg = cfg.augmentation if hasattr(cfg, 'augmentation') else None
    if aug_cfg is not None:
        spatial_enabled = aug_cfg.spatial.enabled
        intensity_enabled = aug_cfg.intensity.enabled
        if spatial_enabled or intensity_enabled:
            print(f"Augmentation enabled: spatial={spatial_enabled}, intensity={intensity_enabled}")
        else:
            print("Augmentation config present but all transforms disabled")
    else:
        print("No augmentation configured (using baseline)")

    per_side_context_slices = int(
        OmegaConf.select(cfg, "data_mode.per_side_context_slices", default=0) or 0
    )
    channel_layout = str(
        OmegaConf.select(cfg, "data_mode.channel_layout", default="slice_major") or "slice_major"
    )
    data_root = cfg.data_io.paths.data_root
    split_file = cfg.data_io.paths.split_file
    preprocessing_configs = OmegaConf.select(cfg, "dataset.preprocessing_configs")

    if loader_mode == "nnunet_slices_2d":
        if capabilities.loader_module not in {_ISLES24_LOADER_MODULE, _ISLES26_LOADER_MODULE}:
            raise NotImplementedError(
                "nnunet_slices_2d dispatch is currently available only for "
                "src.data.loader_stack.isles24_loader / isles26_loader routes."
            )
        shared_cache = {} if cfg.data_runtime.use_shared_cache else None
        cache_lock = threading.Lock() if shared_cache else None

        train_dataset = ISLES24NNUNet2D(
            nnunet_root=cfg.data_io.paths.nnunet_root,
            dataset_id=cfg.dataset.nnunet.dataset_id,
            dataset_name=cfg.dataset.nnunet.dataset_name,
            modalities=cfg.dataset.modalities,
            test_flag=False,
            image_size=cfg.model.image_size,
            transform=None,
            use_caching=cfg.data_runtime.use_caching,
            shared_cache=shared_cache,
            cache_lock=cache_lock,
            aug_cfg=aug_cfg,
            is_training=True,
            per_side_context_slices=per_side_context_slices,
            channel_layout=channel_layout,
        )
        test_dataset = ISLES24NNUNet2D(
            nnunet_root=cfg.data_io.paths.nnunet_root,
            dataset_id=cfg.dataset.nnunet.dataset_id,
            dataset_name=cfg.dataset.nnunet.dataset_name,
            modalities=cfg.dataset.modalities,
            test_flag=True,
            image_size=cfg.model.image_size,
            transform=None,
            use_caching=cfg.data_runtime.use_caching,
            shared_cache=shared_cache,
            cache_lock=cache_lock,
            aug_cfg=None,
            is_training=False,
            per_side_context_slices=per_side_context_slices,
            channel_layout=channel_layout,
        )

    elif loader_mode == "online_slices_3d_to_2d":
        if capabilities.loader_module == _ISLES24_LOADER_MODULE:
            online_dataset_cls = ISLES24OnlineProc2D
        elif capabilities.loader_module == _ISLES26_LOADER_MODULE:
            online_dataset_cls = ISLES26OnlineProc2D
        else:
            raise NotImplementedError(
                "Unsupported loader module for online_slices_3d_to_2d dispatch: "
                f"{capabilities.loader_module}"
            )
        shared_cache = {} if cfg.data_runtime.use_shared_cache else None
        cache_lock = threading.Lock() if shared_cache else None

        train_dataset = online_dataset_cls(
            directory=data_root,
            datalist_json=split_file,
            fold=cfg.dataset.fold,
            transform=None,
            modalities=cfg.dataset.modalities,
            test_flag=False,
            image_size=cfg.model.image_size,
            use_caching=cfg.data_runtime.use_caching,
            shared_cache=shared_cache,
            cache_lock=cache_lock,
            aug_cfg=aug_cfg,
            is_training=True,
            preprocessing_configs=preprocessing_configs,
        )
        test_dataset = online_dataset_cls(
            directory=data_root,
            datalist_json=split_file,
            fold=cfg.dataset.fold,
            transform=None,
            modalities=cfg.dataset.modalities,
            test_flag=True,
            image_size=cfg.model.image_size,
            use_caching=cfg.data_runtime.use_caching,
            shared_cache=shared_cache,
            cache_lock=cache_lock,
            aug_cfg=None,
            is_training=False,
            preprocessing_configs=preprocessing_configs,
        )
    elif loader_mode == "full_volumes_3d":
        if capabilities.loader_module == _ISLES24_LOADER_MODULE:
            fullvol_dataset_cls = ISLES24Dataset3D
        elif capabilities.loader_module == _ISLES26_LOADER_MODULE:
            fullvol_dataset_cls = ISLES26Dataset3D
        else:
            raise NotImplementedError(
                "Unsupported loader module for full_volumes_3d dispatch: "
                f"{capabilities.loader_module}"
            )
        train_dataset = fullvol_dataset_cls(
            directory=data_root,
            datalist_json=split_file,
            fold=cfg.dataset.fold,
            transform=None,
            modalities=cfg.dataset.modalities,
            test_flag=False,
            image_size=cfg.model.image_size,
            preprocessing_configs=preprocessing_configs,
            aug_cfg=aug_cfg,
            is_training=True,
        )
        test_dataset = fullvol_dataset_cls(
            directory=data_root,
            datalist_json=split_file,
            fold=cfg.dataset.fold,
            transform=None,
            modalities=cfg.dataset.modalities,
            test_flag=True,
            image_size=cfg.model.image_size,
            preprocessing_configs=preprocessing_configs,
            aug_cfg=None,
            is_training=False,
        )
    elif loader_mode == "random_patches_3d":
        if capabilities.loader_module == _ISLES24_LOADER_MODULE:
            patch_dataset_cls = ISLES24RandomPatches3D
            fullvol_dataset_cls = ISLES24Dataset3D
        elif capabilities.loader_module == _ISLES26_LOADER_MODULE:
            patch_dataset_cls = ISLES26RandomPatches3D
            fullvol_dataset_cls = ISLES26Dataset3D
        else:
            raise NotImplementedError(
                "Unsupported loader module for random_patches_3d dispatch: "
                f"{capabilities.loader_module}"
            )
        train_dataset = patch_dataset_cls(
            directory=data_root,
            datalist_json=split_file,
            fold=cfg.dataset.fold,
            transform=None,
            modalities=cfg.dataset.modalities,
            test_flag=False,
            image_size=cfg.model.image_size,
            preprocessing_configs=preprocessing_configs,
            aug_cfg=aug_cfg,
            is_training=True,
        )
        # Random patch mode trains on patches but validates/samples on full volumes.
        test_dataset = fullvol_dataset_cls(
            directory=data_root,
            datalist_json=split_file,
            fold=cfg.dataset.fold,
            transform=None,
            modalities=cfg.dataset.modalities,
            test_flag=True,
            image_size=cfg.model.image_size,
            preprocessing_configs=preprocessing_configs,
            aug_cfg=None,
            is_training=False,
        )
    else:
        raise ValueError(f"Unsupported loader_mode: {loader_mode}")

    strategy = resolve_strategy(cfg)
    global_train_batch_size, per_rank_train_batch_size = resolve_train_batch_sizes(
        int(cfg.data_runtime.train_batch_size),
        strategy=strategy,
    )
    print(
        f"Train batch semantics: global={global_train_batch_size}, "
        f"per_rank={per_rank_train_batch_size}, strategy={strategy}"
    )

    val_dataset = test_dataset
    if bool(OmegaConf.select(cfg, "data_runtime.loader_smoke_testing", default=False)):
        train_limit = min(_LOADER_SMOKE_ITEMS, len(train_dataset))
        val_limit = min(_LOADER_SMOKE_ITEMS, len(test_dataset))
        if train_limit == 0:
            raise ValueError(
                "data_runtime.loader_smoke_testing=true but train dataset is empty "
                "(0 items after route/fold split)."
            )
        if val_limit == 0:
            raise ValueError(
                "data_runtime.loader_smoke_testing=true but validation dataset is empty "
                "(0 items after route/fold split)."
            )

        train_dataset = Subset(train_dataset, list(range(train_limit)))
        val_dataset = Subset(test_dataset, list(range(val_limit)))
        print(
            "[Data Runtime] loader_smoke_testing enabled: "
            f"train_items={train_limit}, val_items={val_limit}, "
            f"hard_cap={_LOADER_SMOKE_ITEMS}"
        )

    train_sampler = None
    is_ddp = strategy == "ddp"
    if is_ddp:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError(
                "distribution=ddp requires an initialized torch.distributed process group "
                "before dataloader construction."
            )
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=True,
        )

    train_kwargs = _build_loader_kwargs(
        num_workers=int(cfg.data_runtime.num_train_workers),
        pin_memory=bool(OmegaConf.select(cfg, "data_runtime.pin_memory.train", default=False)),
        persistent_workers=bool(OmegaConf.select(cfg, "data_runtime.persistent_workers.train", default=False)),
        prefetch_factor=OmegaConf.select(cfg, "data_runtime.train_prefetch_factor", default=None),
    )
    val_kwargs = _build_loader_kwargs(
        num_workers=int(cfg.data_runtime.num_valid_workers),
        pin_memory=bool(OmegaConf.select(cfg, "data_runtime.pin_memory.val", default=False)),
        persistent_workers=bool(OmegaConf.select(cfg, "data_runtime.persistent_workers.val", default=False)),
        prefetch_factor=OmegaConf.select(cfg, "data_runtime.test_prefetch_factor", default=None),
    )
    sample_kwargs = _build_loader_kwargs(
        num_workers=int(cfg.data_runtime.num_test_workers),
        pin_memory=bool(OmegaConf.select(cfg, "data_runtime.pin_memory.test", default=False)),
        persistent_workers=bool(OmegaConf.select(cfg, "data_runtime.persistent_workers.test", default=False)),
        prefetch_factor=OmegaConf.select(cfg, "data_runtime.test_prefetch_factor", default=None),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=per_rank_train_batch_size,
        shuffle=not is_ddp,
        sampler=train_sampler,
        **train_kwargs,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.validation.val_batch_size,
        shuffle=False,
        collate_fn=_meta_safe_collate,
        **val_kwargs,
    )
    sample_dataloader = DataLoader(
        test_dataset,
        batch_size=int(cfg.data_runtime.test_batch_size),
        shuffle=True,
        collate_fn=_meta_safe_collate,
        **sample_kwargs,
    )

    return {
        'train': train_dataloader,
        'val': val_dataloader,
        'sample': sample_dataloader
    }


__all__ = [
    "datafold_read",
    "ISLES24Dataset3D",
    "ISLES24RandomPatches3D",
    "ISLES24Dataset2D",
    "ISLES24NNUNet2D",
    "ISLES24OnlineProc2D",
    "ISLES26Dataset3D",
    "ISLES26RandomPatches3D",
    "ISLES26Dataset2D",
    "ISLES26OnlineProc2D",
    "validate_dataset_contract",
    "get_dataloaders",
]
