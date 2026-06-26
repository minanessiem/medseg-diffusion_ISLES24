import unittest

from tests import test_data_contract_generalization as _test_data_contract_generalization
from tests import test_isles26_dataset2d as _test_isles26_dataset2d
from tests import test_isles26_dataset3d as _test_isles26_dataset3d
from tests import test_isles26_dataset_randompatch3d as _test_isles26_dataset_randompatch3d
from tests import test_isles26_random_patch_pipeline as _test_isles26_random_patch_pipeline
from tests import test_loader_stack_routing as _test_loader_stack_routing
from tests import test_loaders_facade_isles26_routing as _test_loaders_facade_isles26_routing


_MODULES = [
    _test_isles26_dataset3d,
    _test_isles26_dataset2d,
    _test_isles26_random_patch_pipeline,
    _test_isles26_dataset_randompatch3d,
    _test_loaders_facade_isles26_routing,
    _test_loader_stack_routing,
    _test_data_contract_generalization,
]


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for module in _MODULES:
        suite.addTests(loader.loadTestsFromModule(module))
    return suite


if __name__ == "__main__":
    unittest.main()
