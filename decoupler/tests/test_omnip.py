import pytest
import pandas as pd
from ..omnip import check_if_omnipath, get_progeny, get_resource, show_resources, get_dorothea


def test_check_if_omnipath():
    check_if_omnipath()


def test_get_progeny():
    df = get_progeny(organism='human', top=100)
    n_paths = len(df['source'].unique())
    n_rows = (n_paths * 100)
    assert type(df) is pd.DataFrame
    assert df.shape[0] == n_rows
    with pytest.raises(ValueError):
        get_progeny(organism='asdfgh')
    get_progeny(organism='mouse')


def test_get_resource():
    res = get_resource('TFcensus')
    assert type(res) is pd.DataFrame
    assert res.shape[0] > 0


def test_show_resources():
    lst = show_resources()
    assert type(lst) is list
    assert len(lst) > 0


def test_get_dorothea():
    df = get_dorothea(organism='human')
    assert type(df) is pd.DataFrame
    assert df.shape[0] > 0
    with pytest.raises(ValueError):
        get_dorothea(organism='asdfgh')
    get_dorothea(organism='mouse')
