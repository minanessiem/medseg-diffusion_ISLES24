import sys
import types
import unittest

import pandas as pd

try:
    import nibabel  # noqa: F401
except ImportError:
    sys.modules["nibabel"] = types.ModuleType("nibabel")

try:
    import tqdm  # noqa: F401
except ImportError:
    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable, **_kwargs: iterable
    sys.modules["tqdm"] = tqdm_module

from scripts.dataset_setup.ISLES26_json_creator import assign_site_balanced_validation_pool


def _metadata_with_singleton_site() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "caseID": ["a1", "a2", "a3", "b1", "c1", "c2"],
            "SITE": ["A", "A", "A", "B", "C", "C"],
        }
    )


class TestSiteBalancedValidationPool(unittest.TestCase):
    def test_singleton_site_stays_in_training(self) -> None:
        result = assign_site_balanced_validation_pool(
            _metadata_with_singleton_site(),
            target_val_full_count=2,
            seed=42,
        )

        self.assertEqual(result.loc[result["SITE"] == "B", "split"].tolist(), ["train"])
        self.assertEqual(result.attrs["singleton_training_sites"], ["B"])

        for site in ("A", "C"):
            site_splits = set(result.loc[result["SITE"] == site, "split"])
            self.assertEqual(site_splits, {"train", "validation_pool"})

    def test_minimum_validation_size_counts_only_multi_case_sites(self) -> None:
        with self.assertRaisesRegex(ValueError, "every multi-case site in validation"):
            assign_site_balanced_validation_pool(
                _metadata_with_singleton_site(),
                target_val_full_count=1,
            )

    def test_maximum_validation_size_reserves_training_cases_and_singletons(self) -> None:
        with self.assertRaisesRegex(ValueError, "singleton sites assigned to training"):
            assign_site_balanced_validation_pool(
                _metadata_with_singleton_site(),
                target_val_full_count=4,
            )


if __name__ == "__main__":
    unittest.main()
