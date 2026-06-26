import copy
import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
from omegaconf import OmegaConf

from src.data.loaders import get_dataloaders


def _write_nifti(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array.astype(np.float32), affine=np.eye(4)), str(path))


def _default_preprocessing_configs() -> dict:
    return {
        "common": {
            "orientation": {"enabled": False, "axcodes": "RAS"},
            "spacing": {
                "enabled": False,
                "allow_native_spacing": True,
                "pixdim": [1.0, 1.0, 1.0],
                "interpolation": {"image": "bilinear", "label": "nearest"},
            },
        },
        "roi": {"volume_3d": [4, 4, 4], "slice_2d": [4, 4]},
        "online_slices_3d_to_2d": {
            "slice_axis": 2,
            "slice_order": "sequential",
        },
        "random_patches_3d": {
            "patches_per_volume": {"train": 2},
            "rand_crop_by_pos_neg_label": {
                "pos": 1,
                "neg": 1,
                "image_threshold": 0.0,
                "allow_smaller": True,
            },
            "pad_to_divisible": True,
        },
        "full_volumes_3d": {"pad_to_divisible": False},
    }


def _build_cfg(
    data_root: str,
    split_file: str,
    loader_mode: str,
    dim: str,
    preprocessing_configs=None,
):
    if preprocessing_configs is None:
        preprocessing_configs = _default_preprocessing_configs()
    return OmegaConf.create(
        {
            "dataset": {
                "id": "isles26",
                "name": "isles26",
                "fold": 0,
                "partitioning": "split",
                "subsets": {
                    "train": {"split_in": ["train"]},
                    "val_fast": {"split_in": ["val_fast"]},
                    "val_rest": {"split_in": ["val_rest"]},
                    "val_full": {"split_in": ["val_fast", "val_rest"]},
                },
                "active_subsets": {
                    "train": "train",
                    "val": "val_fast",
                    "sample": "val_fast",
                },
                "modalities": ["T1_RAW"],
                "num_modalities": 1,
                "nnunet": {"dataset_id": "501", "dataset_name": "isles26"},
                "preprocessing_configs": preprocessing_configs,
            },
            "data_mode": {
                "loader_mode": loader_mode,
                "dim": dim,
                "per_side_context_slices": 0,
                "channel_layout": None,
            },
            "data_io": {
                "paths": {
                    "data_root": data_root,
                    "split_file": split_file,
                    "nnunet_root": "/tmp/nnunet",
                }
            },
            "data_runtime": {
                "train_batch_size": 1,
                "test_batch_size": 1,
                "num_train_workers": 0,
                "num_valid_workers": 0,
                "num_test_workers": 0,
                "use_caching": False,
                "use_shared_cache": False,
                "train_prefetch_factor": 2,
                "test_prefetch_factor": 2,
                "loader_smoke_testing": False,
            },
            "model": {"image_size": 4},
            "validation": {"val_batch_size": 1},
        }
    )


class TestLoadersFacadeIsles26Routing(unittest.TestCase):
    def _prepare_isles26_split(self, base: Path):
        case_a_t1 = np.stack(
            [np.full((4, 4), 1.0, dtype=np.float32), np.full((4, 4), 2.0, dtype=np.float32)],
            axis=-1,
        )
        case_a_label = np.stack(
            [np.full((4, 4), 0.0, dtype=np.float32), np.full((4, 4), 1.0, dtype=np.float32)],
            axis=-1,
        )
        case_b_t1 = np.stack(
            [np.full((4, 4), 10.0, dtype=np.float32), np.full((4, 4), 11.0, dtype=np.float32)],
            axis=-1,
        )
        case_b_label = np.stack(
            [np.full((4, 4), 1.0, dtype=np.float32), np.full((4, 4), 2.0, dtype=np.float32)],
            axis=-1,
        )
        case_c_t1 = np.stack(
            [np.full((4, 4), 20.0, dtype=np.float32), np.full((4, 4), 21.0, dtype=np.float32)],
            axis=-1,
        )
        case_c_label = np.stack(
            [np.full((4, 4), 2.0, dtype=np.float32), np.full((4, 4), 3.0, dtype=np.float32)],
            axis=-1,
        )

        _write_nifti(base / "R001/sub-r001s001/ses-1/anat/t1_a.nii.gz", case_a_t1)
        _write_nifti(base / "R001/sub-r001s001/ses-1/anat/label_a.nii.gz", case_a_label)
        _write_nifti(base / "R002/sub-r002s001/ses-1/anat/t1_b.nii.gz", case_b_t1)
        _write_nifti(base / "R002/sub-r002s001/ses-1/anat/label_b.nii.gz", case_b_label)
        _write_nifti(base / "R003/sub-r003s001/ses-1/anat/t1_c.nii.gz", case_c_t1)
        _write_nifti(base / "R003/sub-r003s001/ses-1/anat/label_c.nii.gz", case_c_label)

        datalist = {
            "training": [
                {
                    "split": "val_fast",
                    "caseID": "sub-r001s001",
                    "T1": ["R001/sub-r001s001/ses-1/anat/t1_a.nii.gz"],
                    "label": "R001/sub-r001s001/ses-1/anat/label_a.nii.gz",
                },
                {
                    "split": "train",
                    "caseID": "sub-r002s001",
                    "T1": ["R002/sub-r002s001/ses-1/anat/t1_b.nii.gz"],
                    "label": "R002/sub-r002s001/ses-1/anat/label_b.nii.gz",
                },
                {
                    "split": "val_rest",
                    "caseID": "sub-r003s001",
                    "T1": ["R003/sub-r003s001/ses-1/anat/t1_c.nii.gz"],
                    "label": "R003/sub-r003s001/ses-1/anat/label_c.nii.gz",
                },
            ]
        }
        datalist_path = base / "isles26.json"
        datalist_path.write_text(json.dumps(datalist), encoding="utf-8")
        return datalist_path

    def _prepare_isles26_nnunet_dataset(self, base: Path):
        dataset_dir = base / "Dataset260_isles26_t1_raw"
        images_tr = dataset_dir / "imagesTr"
        labels_tr = dataset_dir / "labelsTr"
        images_ts = dataset_dir / "imagesTs"
        labels_ts = dataset_dir / "labelsTs"

        _write_nifti(images_tr / "sub-r001s001_s0000_0000.nii.gz", np.ones((4, 4), dtype=np.float32))
        _write_nifti(labels_tr / "sub-r001s001_s0000.nii.gz", np.zeros((4, 4), dtype=np.float32))

        _write_nifti(images_ts / "sub-r002s001_s0000_0000.nii.gz", np.full((4, 4), 2.0, dtype=np.float32))
        _write_nifti(labels_ts / "sub-r002s001_s0000.nii.gz", np.ones((4, 4), dtype=np.float32))
        return dataset_dir

    def test_get_dataloaders_routes_isles26_online_2d(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            datalist_path = self._prepare_isles26_split(base)
            cfg = _build_cfg(
                data_root=str(base),
                split_file=str(datalist_path),
                loader_mode="online_slices_3d_to_2d",
                dim="2d",
            )

            dataloaders = get_dataloaders(cfg)
            train_batch = next(iter(dataloaders["train"]))
            val_batch = next(iter(dataloaders["val"]))

            self.assertEqual(train_batch[0].shape[1:], (1, 4, 4))
            self.assertEqual(val_batch[0].shape[1:], (1, 4, 4))

    def test_get_dataloaders_isles26_online_2d_supports_slice_axis_order_and_roi_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            datalist_path = self._prepare_isles26_split(base)
            preprocessing_configs = copy.deepcopy(_default_preprocessing_configs())
            preprocessing_configs["online_slices_3d_to_2d"]["slice_axis"] = 0
            preprocessing_configs["online_slices_3d_to_2d"]["slice_order"] = "reverse"
            preprocessing_configs["roi"]["slice_2d"] = [6, 5]

            cfg = _build_cfg(
                data_root=str(base),
                split_file=str(datalist_path),
                loader_mode="online_slices_3d_to_2d",
                dim="2d",
                preprocessing_configs=preprocessing_configs,
            )

            dataloaders = get_dataloaders(cfg)
            val_batch = next(iter(dataloaders["val"]))
            self.assertEqual(val_batch[0].shape[1:], (1, 6, 5))
            self.assertEqual(val_batch[2][0], "sub-r001s001_slice3")

    def test_get_dataloaders_routes_isles26_fullvol_3d(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            datalist_path = self._prepare_isles26_split(base)
            cfg = _build_cfg(
                data_root=str(base),
                split_file=str(datalist_path),
                loader_mode="full_volumes_3d",
                dim="3d",
            )

            dataloaders = get_dataloaders(cfg)
            train_batch = next(iter(dataloaders["train"]))
            val_batch = next(iter(dataloaders["val"]))

            self.assertEqual(train_batch[0].shape[1:], (1, 4, 4, 2))
            self.assertEqual(val_batch[0].shape[1:], (1, 4, 4, 2))

    def test_get_dataloaders_routes_3d_augmentation_to_train_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            case_a_t1 = np.arange(4 * 4 * 2, dtype=np.float32).reshape(4, 4, 2)
            case_a_label = np.zeros((4, 4, 2), dtype=np.float32)
            case_a_label[1, 1, 1] = 1.0
            case_b_t1 = np.arange(4 * 4 * 2, dtype=np.float32).reshape(4, 4, 2) + 100.0
            case_b_label = np.zeros((4, 4, 2), dtype=np.float32)
            case_b_label[1, 1, 1] = 1.0

            _write_nifti(base / "R001/sub-r001s001/ses-1/anat/t1_a.nii.gz", case_a_t1)
            _write_nifti(base / "R001/sub-r001s001/ses-1/anat/label_a.nii.gz", case_a_label)
            _write_nifti(base / "R002/sub-r002s001/ses-1/anat/t1_b.nii.gz", case_b_t1)
            _write_nifti(base / "R002/sub-r002s001/ses-1/anat/label_b.nii.gz", case_b_label)

            datalist = {
                "training": [
                    {
                        "split": "val_fast",
                        "caseID": "sub-r001s001",
                        "T1": ["R001/sub-r001s001/ses-1/anat/t1_a.nii.gz"],
                        "label": "R001/sub-r001s001/ses-1/anat/label_a.nii.gz",
                    },
                    {
                        "split": "train",
                        "caseID": "sub-r002s001",
                        "T1": ["R002/sub-r002s001/ses-1/anat/t1_b.nii.gz"],
                        "label": "R002/sub-r002s001/ses-1/anat/label_b.nii.gz",
                    },
                ]
            }
            datalist_path = base / "isles26.json"
            datalist_path.write_text(json.dumps(datalist), encoding="utf-8")

            cfg = _build_cfg(
                data_root=str(base),
                split_file=str(datalist_path),
                loader_mode="full_volumes_3d",
                dim="3d",
            )
            cfg.augmentation = OmegaConf.create(
                {
                    "spatial": {
                        "enabled": True,
                        "random_flip": {
                            "enabled": True,
                            "prob": 1.0,
                            "spatial_axis": [0],
                        },
                    },
                    "intensity": {"enabled": False},
                }
            )

            dataloaders = get_dataloaders(cfg)
            train_batch = next(iter(dataloaders["train"]))
            val_batch = next(iter(dataloaders["val"]))

            self.assertAlmostEqual(
                float(train_batch[0][0, 0, 0, 0, 0].item()),
                float(case_b_t1[-1, 0, 0]),
            )
            self.assertAlmostEqual(
                float(val_batch[0][0, 0, 0, 0, 0].item()),
                float(case_a_t1[0, 0, 0]),
            )

    def test_get_dataloaders_isles26_fullvol_3d_supports_padding_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            datalist_path = self._prepare_isles26_split(base)
            preprocessing_configs = copy.deepcopy(_default_preprocessing_configs())
            preprocessing_configs["full_volumes_3d"]["pad_to_divisible"] = True
            preprocessing_configs["roi"]["volume_3d"] = [6, 5, 4]

            cfg = _build_cfg(
                data_root=str(base),
                split_file=str(datalist_path),
                loader_mode="full_volumes_3d",
                dim="3d",
                preprocessing_configs=preprocessing_configs,
            )

            dataloaders = get_dataloaders(cfg)
            val_batch = next(iter(dataloaders["val"]))
            self.assertEqual(val_batch[0].shape[1:], (1, 6, 5, 4))
            self.assertEqual(val_batch[1].shape[1:], (1, 6, 5, 4))

    def test_get_dataloaders_isles26_random_patch_train_and_full_volume_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            datalist_path = self._prepare_isles26_split(base)
            preprocessing_configs = copy.deepcopy(_default_preprocessing_configs())
            preprocessing_configs["roi"]["volume_3d"] = [4, 4, 4]
            preprocessing_configs["random_patches_3d"]["patches_per_volume"]["train"] = 2
            preprocessing_configs["random_patches_3d"]["pad_to_divisible"] = True

            cfg = _build_cfg(
                data_root=str(base),
                split_file=str(datalist_path),
                loader_mode="random_patches_3d",
                dim="3d",
                preprocessing_configs=preprocessing_configs,
            )

            dataloaders = get_dataloaders(cfg)
            train_batch = next(iter(dataloaders["train"]))
            val_batch = next(iter(dataloaders["val"]))
            sample_batch = next(iter(dataloaders["sample"]))

            self.assertEqual(train_batch[0].shape[1:], (1, 4, 4, 4))
            self.assertIn("_patch", train_batch[2][0])

            self.assertEqual(val_batch[0].shape[1:], (1, 4, 4, 2))
            self.assertEqual(sample_batch[0].shape[1:], (1, 4, 4, 2))
            self.assertEqual(val_batch[2][0], "sub-r001s001")

    def test_get_dataloaders_isles26_fullvol_supports_val_full_union_subset(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            datalist_path = self._prepare_isles26_split(base)
            cfg = _build_cfg(
                data_root=str(base),
                split_file=str(datalist_path),
                loader_mode="full_volumes_3d",
                dim="3d",
            )
            cfg.dataset.active_subsets.val = "val_full"
            cfg.dataset.active_subsets.sample = "val_full"

            dataloaders = get_dataloaders(cfg)

            self.assertEqual(len(dataloaders["val"].dataset), 2)
            self.assertEqual(len(dataloaders["sample"].dataset), 2)

    def test_get_dataloaders_routes_isles26_nnunet_2d(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self._prepare_isles26_nnunet_dataset(base)
            cfg = _build_cfg(
                data_root="/unused",
                split_file="/unused.json",
                loader_mode="nnunet_slices_2d",
                dim="2d",
            )
            cfg.data_mode.per_side_context_slices = 0
            cfg.data_mode.channel_layout = "slice_major"
            cfg.data_io.paths.nnunet_root = str(base)
            cfg.dataset.nnunet.dataset_id = "260"
            cfg.dataset.nnunet.dataset_name = "isles26_t1_raw"

            dataloaders = get_dataloaders(cfg)
            train_batch = next(iter(dataloaders["train"]))
            val_batch = next(iter(dataloaders["val"]))

            self.assertEqual(train_batch[0].shape[1:], (1, 4, 4))
            self.assertEqual(val_batch[0].shape[1:], (1, 4, 4))
            self.assertEqual(train_batch[2][0], "sub-r001s001_slice0")
            self.assertEqual(val_batch[2][0], "sub-r002s001_slice0")


if __name__ == "__main__":
    unittest.main()
