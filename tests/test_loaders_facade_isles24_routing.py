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
                "pixdim": [1.0, 1.0, 1.0],
                "interpolation": {"image": "bilinear", "label": "nearest"},
            },
        },
        "roi": {"volume_3d": [4, 4, 4], "slice_2d": [4, 4]},
        "online_slices_3d_to_2d": {"slice_axis": 2, "slice_order": "sequential"},
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
                "id": "isles24",
                "name": "isles24",
                "fold": 0,
                "partitioning": "fold",
                "subsets": {
                    "train": {"fold_not_in": [0]},
                    "val": {"fold_in": [0]},
                },
                "active_subsets": {
                    "train": "train",
                    "val": "val",
                    "sample": "val",
                },
                "modalities": ["CBF_min_0_max_70", "TMAX_min_4_max_30"],
                "num_modalities": 2,
                "nnunet": {"dataset_id": "050", "dataset_name": "isles24_baseline"},
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


class TestLoadersFacadeIsles24Routing(unittest.TestCase):
    def _prepare_isles24_split(self, base: Path):
        shape = (4, 4, 2)
        case_a_cbf = np.full(shape, 20.0, dtype=np.float32)
        case_a_tmax = np.full(shape, 10.0, dtype=np.float32)
        case_b_cbf = np.full(shape, 25.0, dtype=np.float32)
        case_b_tmax = np.full(shape, 12.0, dtype=np.float32)
        label = np.zeros(shape, dtype=np.float32)
        label[1, 1, 1] = 1.0

        _write_nifti(base / "R001/sub-r001s001/ses-1/anat/cbf_a.nii.gz", case_a_cbf)
        _write_nifti(base / "R001/sub-r001s001/ses-1/anat/tmax_a.nii.gz", case_a_tmax)
        _write_nifti(base / "R001/sub-r001s001/ses-1/anat/label_a.nii.gz", label)
        _write_nifti(base / "R002/sub-r002s001/ses-1/anat/cbf_b.nii.gz", case_b_cbf)
        _write_nifti(base / "R002/sub-r002s001/ses-1/anat/tmax_b.nii.gz", case_b_tmax)
        _write_nifti(base / "R002/sub-r002s001/ses-1/anat/label_b.nii.gz", label)

        datalist = {
            "training": [
                {
                    "fold": 0,
                    "caseID": "sub-r001s001",
                    "CBF": ["R001/sub-r001s001/ses-1/anat/cbf_a.nii.gz"],
                    "TMAX": ["R001/sub-r001s001/ses-1/anat/tmax_a.nii.gz"],
                    "label": "R001/sub-r001s001/ses-1/anat/label_a.nii.gz",
                },
                {
                    "fold": 1,
                    "caseID": "sub-r002s001",
                    "CBF": ["R002/sub-r002s001/ses-1/anat/cbf_b.nii.gz"],
                    "TMAX": ["R002/sub-r002s001/ses-1/anat/tmax_b.nii.gz"],
                    "label": "R002/sub-r002s001/ses-1/anat/label_b.nii.gz",
                },
            ]
        }
        datalist_path = base / "isles24.json"
        datalist_path.write_text(json.dumps(datalist), encoding="utf-8")
        return datalist_path

    def test_get_dataloaders_isles24_random_patch_train_and_full_volume_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            datalist_path = self._prepare_isles24_split(base)
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

            self.assertEqual(train_batch[0].shape[1:], (2, 4, 4, 4))
            self.assertIn("_patch", train_batch[2][0])

            self.assertEqual(val_batch[0].shape[1:], (2, 4, 4, 2))
            self.assertEqual(sample_batch[0].shape[1:], (2, 4, 4, 2))
            self.assertEqual(val_batch[2][0], "sub-r001s001")


if __name__ == "__main__":
    unittest.main()
