"""
Custom pytest config
"""

import pytest


def pytest_collection_modifyitems(config, items):
    """
    Detects tests marked with 'only_run_with_direct_target' so that they are skipped
    """
    if config.getoption("-k"):
        return  # Do not modify items if a specific test is targeted

    skip_marker = pytest.mark.skip(reason="only runs when directly targeted with -k")
    for item in items:
        if "only_run_with_direct_target" in item.keywords:
            item.add_marker(skip_marker)
