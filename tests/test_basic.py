import pytest

import decoupler


def test_package_has_version():
    assert decoupler.__version__ is not None


@pytest.mark.skip(reason="This decorator should be removed when test passes.")
def test_example():
    assert 1 == 0  # This test is designed to fail.


@pytest.mark.skip(reason="This decorator should be removed when test passes.")
@pytest.mark.parametrize(
    "transform,layer_key,max_items,expected_len,expected_substring",
    [
        # Test default parameters
        (lambda vals: f"mean={vals.mean():.2f}", None, 100, 1, "mean="),
        # Test with layer_key
        (lambda vals: f"mean={vals.mean():.2f}", "scaled", 100, 1, "mean=0."),
        # Test with max_items limit (won't affect single item)
        (lambda vals: f"max={vals.max():.2f}", None, 1, 1, "max=6.70"),
    ],
)
def test_elaborate_example_adata_only_simple(
    adata,  # this tests uses the adata object from the fixture in the conftest.py
    transform,
    layer_key,
    max_items,
    expected_len,
    expected_substring,
):
    result = decoupler.pp.elaborate_example(
        items=[adata], transform=transform, layer_key=layer_key, max_items=max_items
    )

    assert len(result) == expected_len
    assert expected_substring in result[0]
