import numpy as np
import pandas as pd
import pytest

import decoupler as dc


def test_show_organisms():
    lst = dc.op.show_organisms()
    assert isinstance(lst, list)
    assert len(lst) > 0
    assert {'mouse', 'rat'}.issubset(lst)


@pytest.mark.parametrize(
    'lst,my_dict,one_to_many',
    [
        [['a', 'b', 'c', 'd'], dict(a=['B', 'C'], b=['A', 'C'], c=['A', 'B'], d='D'), 1],
        [['a', 'b', 'c', 'd'], dict(c=['A', 'B']), 1],
        [['a', 'b', 'c', 'd'], dict(a=['B', 'C'], b=['A', 'C'], c=['A', 'B'], d='D'), 10],
    ]
)
def test_replace_subunits(
    lst,
    my_dict,
    one_to_many,
):  
    res = dc.op._translate._replace_subunits(
        lst=lst, my_dict=my_dict, one_to_many=one_to_many
    )
    assert isinstance(res, list)
    assert len(res) == len(lst)
    for k in my_dict:
        idx = lst.index(k)
        if k in my_dict:
            if len(my_dict[k]) > one_to_many:
                assert np.isnan(res[idx])
            else:
                assert isinstance(res[idx], list)
        else:
            assert np.isnan(res[idx])


@pytest.mark.parametrize('target_organism', ['mouse', 'anole_lizard', 'fruitfly'])
def test_translate(
    target_organism,
):
    net = dc.op.collectri()
    t_net = dc.op.translate(net=net, columns='target', target_organism='mouse')
    cols = {'source', 'target', 'weight'}
    assert isinstance(t_net, pd.DataFrame)
    assert cols.issubset(t_net.columns)
    assert net.shape[0] != t_net.shape[0]






