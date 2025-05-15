import pytest

import decoupler


def test_package_has_version():
    assert decoupler.__version__ is not None    
