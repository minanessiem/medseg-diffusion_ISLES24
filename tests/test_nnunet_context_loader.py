import os
import shutil
import tempfile
import unittest

import nibabel as nib
import numpy as np

from src.data.loaders import ISLES24NNUNet2D


class TestNNUNetContextLoader(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="nnunet_ctx_test_")
        self.dataset_id = "050"
        self.dataset_name = "ctxunittest"
        self.volume_id = "caseA"
        self.modalities = ["CBF", "TMAX"]
        self.image_size = 2
        self._create_nnunet_like_dataset()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _create_nnunet_like_dataset(self):
        dataset_dir = os.path.join(
            self.tmp_dir, f"Dataset{self.dataset_id}_{self.dataset_name}"
        )
        images_tr = os.path.join(dataset_dir, "imagesTr")
        labels_tr = os.path.join(dataset_dir, "labelsTr")
        os.makedirs(images_tr, exist_ok=True)
        os.makedirs(labels_tr, exist_ok=True)

        affine = np.eye(4, dtype=np.float32)
        for slice_idx in range(3):
            slice_stem = f"{self.volume_id}_s{slice_idx:04d}"

            # Modality channel values encode channel + slice index for deterministic assertions.
            ch0 = np.full((2, 2), 10 + slice_idx, dtype=np.float32)
            ch1 = np.full((2, 2), 20 + slice_idx, dtype=np.float32)
            nib.save(
                nib.Nifti1Image(ch0, affine),
                os.path.join(images_tr, f"{slice_stem}_0000.nii.gz"),
            )
            nib.save(
                nib.Nifti1Image(ch1, affine),
                os.path.join(images_tr, f"{slice_stem}_0001.nii.gz"),
            )

            label = np.full((2, 2), slice_idx, dtype=np.float32)
            nib.save(
                nib.Nifti1Image(label, affine),
                os.path.join(labels_tr, f"{slice_stem}.nii.gz"),
            )

    def _build_dataset(self, per_side_context_slices, channel_layout):
        return ISLES24NNUNet2D(
            nnunet_root=self.tmp_dir,
            dataset_id=self.dataset_id,
            dataset_name=self.dataset_name,
            modalities=self.modalities,
            test_flag=False,
            image_size=self.image_size,
            transform=None,
            use_caching=False,
            aug_cfg=None,
            is_training=False,
            per_side_context_slices=per_side_context_slices,
            channel_layout=channel_layout,
        )

    def test_k0_preserves_legacy_channel_count(self):
        dataset = self._build_dataset(per_side_context_slices=0, channel_layout="slice_major")
        image, label, _ = dataset[0]
        self.assertEqual(tuple(image.shape), (2, 2, 2))
        self.assertEqual(tuple(label.shape), (1, 2, 2))

    def test_slice_major_boundary_zero_padding(self):
        dataset = self._build_dataset(per_side_context_slices=1, channel_layout="slice_major")
        first_idx = next(i for i, s in enumerate(dataset.samples) if int(s["slice_index"]) == 0)
        image, label, _ = dataset[first_idx]

        self.assertEqual(tuple(image.shape), (6, 2, 2))
        self.assertEqual(tuple(label.shape), (1, 2, 2))

        # slice_major with k=1: [-1:{m0,m1}, 0:{m0,m1}, +1:{m0,m1}]
        observed = [float(image[c, 0, 0].item()) for c in range(6)]
        expected = [0.0, 0.0, 10.0, 20.0, 11.0, 21.0]
        self.assertEqual(observed, expected)

    def test_modality_major_channel_order(self):
        dataset = self._build_dataset(per_side_context_slices=1, channel_layout="modality_major")
        center_idx = next(i for i, s in enumerate(dataset.samples) if int(s["slice_index"]) == 1)
        image, _, _ = dataset[center_idx]

        self.assertEqual(tuple(image.shape), (6, 2, 2))

        # modality_major with k=1: [m0:-1,0,+1, m1:-1,0,+1]
        observed = [float(image[c, 0, 0].item()) for c in range(6)]
        expected = [10.0, 11.0, 12.0, 20.0, 21.0, 22.0]
        self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
