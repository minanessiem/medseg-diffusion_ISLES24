"""
Unit tests for nnU-Net converter affine helpers.
"""

import unittest

import numpy as np

from scripts.nnunet.convert_to_nnunet import _build_export_affine


class TestNnunetConverterAffine(unittest.TestCase):
    def test_build_export_affine_applies_resize_scaling_and_slice_offset(self):
        source_affine = np.array(
            [
                [2.0, 0.0, 0.0, 10.0],
                [0.0, 3.0, 0.0, 20.0],
                [0.0, 0.0, 5.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        slice_meta = {
            "source_affine": source_affine.tolist(),
            "pre_resize_shape_hw": [200, 100],
        }
        # Downsample to 100x50 -> scale factor 2.0 along both in-plane dims.
        affine = _build_export_affine(slice_meta=slice_meta, out_h=100, out_w=50, slice_idx=7)

        self.assertAlmostEqual(float(affine[0, 0]), 4.0)  # 2.0 * (200/100)
        self.assertAlmostEqual(float(affine[1, 1]), 6.0)  # 3.0 * (100/50)
        self.assertAlmostEqual(float(affine[2, 2]), 5.0)  # unchanged slice direction scale
        # z-offset: 30 + 7*5 = 65
        self.assertAlmostEqual(float(affine[2, 3]), 65.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
