"""
Round-trip tests for nnU-Net slice export geometry/provenance.
"""

import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from scripts.nnunet.convert_to_nnunet import (
    process_single_slice,
    write_provenance_jsonl,
)


class _SyntheticSliceDataset:
    """
    Minimal dataset stub returning metadata-enabled slice samples.
    """

    def __init__(self):
        self._source_affine = np.array(
            [
                [1.5, 0.0, 0.0, 10.0],
                [0.0, 2.0, 0.0, 20.0],
                [0.0, 0.0, 4.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self._case_id = "sub-stroke9999"

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        image = torch.zeros((2, 4, 5), dtype=torch.float32)
        image[0, :, :] = float(idx)  # channel 0 constant == slice index
        image[1, :, :] = float(idx + 100)  # channel 1 offset constant

        label = torch.zeros((1, 4, 5), dtype=torch.float32)
        label[0, 1:3, 2:4] = float(idx + 1)

        virtual_path = f"{self._case_id}_slice{idx}"
        slice_meta = {
            "case_id": self._case_id,
            "slice_index": idx,
            "virtual_path": virtual_path,
            "source_path": f"/synthetic/{self._case_id}.nii.gz",
            "source_affine": self._source_affine.tolist(),
            "source_spacing_xyz": [1.5, 2.0, 4.0],
            "source_axcodes": ["R", "A", "S"],
            "source_volume_shape": [4, 5, 3],
            "slice_axis": 2,
            "pre_resize_shape_hw": [4, 5],
            "post_resize_shape_hw": [4, 5],
        }
        return image, label, virtual_path, slice_meta


class TestNnunetConverterRoundTrip(unittest.TestCase):
    def test_roundtrip_reconstruction_and_affine_offsets(self):
        dataset = _SyntheticSliceDataset()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            images_dir = base / "images"
            labels_dir = base / "labels"
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)

            records = []
            for idx in range(len(dataset)):
                record = process_single_slice(
                    idx=idx,
                    dataset=dataset,
                    images_dir=images_dir,
                    labels_dir=labels_dir,
                    num_channels=2,
                    split_name="test",
                )
                records.append(record)

            # Validate affine z-offset progression by exported label files.
            z_translations = []
            reconstructed_slices = []
            for idx in range(len(dataset)):
                label_path = labels_dir / f"sub-stroke9999_s{idx:04d}.nii.gz"
                self.assertTrue(label_path.exists())
                nii = nib.load(label_path)
                z_translations.append(float(nii.affine[2, 3]))
                reconstructed_slices.append(nii.get_fdata().squeeze(-1))

            self.assertEqual(z_translations, [30.0, 34.0, 38.0])

            # Reconstruct [H, W, D] and verify per-slice label values.
            reconstructed_volume = np.stack(reconstructed_slices, axis=-1)
            self.assertEqual(reconstructed_volume.shape, (4, 5, 3))
            # Center ROI values should be 1,2,3 across slice axis.
            roi_values = reconstructed_volume[1:3, 2:4, :]
            self.assertTrue(np.all(roi_values[:, :, 0] == 1.0))
            self.assertTrue(np.all(roi_values[:, :, 1] == 2.0))
            self.assertTrue(np.all(roi_values[:, :, 2] == 3.0))

            # Validate JSONL provenance completeness.
            jsonl_path = base / "export_provenance.jsonl"
            write_provenance_jsonl(records, jsonl_path)
            self.assertTrue(jsonl_path.exists())
            with jsonl_path.open("r", encoding="utf-8") as handle:
                parsed = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(len(parsed), 3)
            self.assertEqual(parsed[0]["safe_case_id"], "sub-stroke9999_s0000")
            self.assertEqual(parsed[2]["slice_index"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
