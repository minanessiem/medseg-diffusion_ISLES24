import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from src.data.loader_stack.isles26_loader import build_random_patches_3d_pipeline


def _write_nifti(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array.astype(np.float32), affine=np.eye(4)), str(path))


def _build_preprocessing_configs(
    *,
    roi_3d: list[int],
    patches_per_volume: int,
    pad_to_divisible: bool,
) -> dict:
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
        "roi": {
            "volume_3d": roi_3d,
            "slice_2d": [roi_3d[0], roi_3d[1]],
        },
        "online_slices_3d_to_2d": {"slice_axis": 2, "slice_order": "sequential"},
        "random_patches_3d": {
            "patches_per_volume": {"train": patches_per_volume},
            "rand_crop_by_pos_neg_label": {
                "pos": 1,
                "neg": 1,
                "image_threshold": 0.0,
                "allow_smaller": True,
            },
            "pad_to_divisible": pad_to_divisible,
        },
        "full_volumes_3d": {"pad_to_divisible": False},
    }


def _as_sample_list(output):
    if isinstance(output, list):
        return output
    return [output]


class TestIsles26RandomPatchPipeline(unittest.TestCase):
    def test_random_patch_pipeline_respects_num_samples_and_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            t1_path = base / "R001/sub-r001s001/ses-1/anat/t1.nii.gz"
            label_path = base / "R001/sub-r001s001/ses-1/anat/label.nii.gz"

            image = np.ones((3, 4, 2), dtype=np.float32) * 5.0
            label = np.zeros((3, 4, 2), dtype=np.float32)
            label[1, 1, 1] = 1.0
            _write_nifti(t1_path, image)
            _write_nifti(label_path, label)

            preprocessing_configs = _build_preprocessing_configs(
                roi_3d=[6, 5, 4],
                patches_per_volume=2,
                pad_to_divisible=True,
            )
            pipeline = build_random_patches_3d_pipeline(
                modalities=["T1_RAW"],
                preprocessing_configs=preprocessing_configs,
            )

            outputs = _as_sample_list(
                pipeline({"T1": str(t1_path), "label": str(label_path)})
            )
            self.assertEqual(len(outputs), 2)
            for sample in outputs:
                self.assertEqual(tuple(sample["image"].shape), (1, 6, 5, 4))
                self.assertEqual(tuple(sample["label"].shape), (1, 6, 5, 4))

    def test_random_patch_pipeline_allows_smaller_without_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            t1_path = base / "R001/sub-r001s001/ses-1/anat/t1.nii.gz"
            label_path = base / "R001/sub-r001s001/ses-1/anat/label.nii.gz"

            image = np.ones((3, 4, 2), dtype=np.float32) * 7.0
            label = np.zeros((3, 4, 2), dtype=np.float32)
            label[1, 1, 1] = 1.0
            _write_nifti(t1_path, image)
            _write_nifti(label_path, label)

            preprocessing_configs = _build_preprocessing_configs(
                roi_3d=[6, 5, 4],
                patches_per_volume=1,
                pad_to_divisible=False,
            )
            pipeline = build_random_patches_3d_pipeline(
                modalities=["T1_RAW"],
                preprocessing_configs=preprocessing_configs,
            )

            outputs = _as_sample_list(
                pipeline({"T1": str(t1_path), "label": str(label_path)})
            )
            self.assertEqual(len(outputs), 1)
            sample = outputs[0]
            self.assertEqual(tuple(sample["image"].shape), (1, 3, 4, 2))
            self.assertEqual(tuple(sample["label"].shape), (1, 3, 4, 2))


if __name__ == "__main__":
    unittest.main()
