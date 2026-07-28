import unittest

from src.data.loader_stack.factory import resolve_loader_contract
from src.data.loader_stack.registry import get_dataset_capabilities


class TestLoaderStackRouting(unittest.TestCase):
    def test_get_dataset_capabilities_known_dataset(self):
        capabilities = get_dataset_capabilities("isles24")
        self.assertEqual(capabilities.dataset_id, "isles24")
        self.assertIn("nnunet_slices_2d", capabilities.supported_loader_modes)

    def test_get_dataset_capabilities_unknown_dataset_fails(self):
        with self.assertRaisesRegex(ValueError, "Unsupported dataset id"):
            get_dataset_capabilities("unknown-dataset")

    def test_resolve_loader_contract_allows_supported_combination(self):
        resolution = resolve_loader_contract(
            dataset_id="isles24",
            dataset_name=None,
            loader_mode="nnunet_slices_2d",
        )
        self.assertEqual(resolution.dataset_id, "isles24")
        self.assertEqual(resolution.loader_mode, "nnunet_slices_2d")

    def test_resolve_loader_contract_allows_isles24_random_patches_3d(self):
        resolution = resolve_loader_contract(
            dataset_id="isles24",
            dataset_name=None,
            loader_mode="random_patches_3d",
        )
        self.assertEqual(resolution.dataset_id, "isles24")
        self.assertEqual(resolution.loader_mode, "random_patches_3d")

    def test_resolve_loader_contract_allows_isles26_random_patches_3d(self):
        resolution = resolve_loader_contract(
            dataset_id="isles26",
            dataset_name=None,
            loader_mode="random_patches_3d",
        )
        self.assertEqual(resolution.dataset_id, "isles26")
        self.assertEqual(resolution.loader_mode, "random_patches_3d")

    def test_resolve_loader_contract_allows_isles26_nnunet_slices_2d(self):
        resolution = resolve_loader_contract(
            dataset_id="isles26",
            dataset_name=None,
            loader_mode="nnunet_slices_2d",
        )
        self.assertEqual(resolution.dataset_id, "isles26")
        self.assertEqual(resolution.loader_mode, "nnunet_slices_2d")

    def test_resolve_loader_contract_rejects_invalid_loader_mode(self):
        with self.assertRaisesRegex(ValueError, "Invalid data_mode.loader_mode"):
            resolve_loader_contract(
                dataset_id="isles24",
                dataset_name=None,
                loader_mode="not_a_mode",
            )


if __name__ == "__main__":
    unittest.main()
