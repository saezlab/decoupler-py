import pandas as pd
import numpy as np
import pytest

import decoupler as dc


@pytest.mark.parametrize(
    'names,order,label,nbins',
    [
        [None, 'pstime', None, 10],
        ['G05', 'pstime', None, 35],
        [['G05'], 'f_order', None, 10],
        [['G01', 'G06', 'G10'], 'pstime', 'group', 14],
    ]
)
def test_bin_order(
    tdata,
    names,
    order,
    label,
    nbins,
):
    rng = np.random.default_rng(seed=42)
    tdata.obs.loc[:, 'f_order'] = rng.random(tdata.n_obs)
    df = dc.pp.bin_order(adata=tdata, names=names, order=order, label=label, nbins=nbins)
    assert isinstance(df, pd.DataFrame)
    cols = {'name', 'order', 'value'}
    assert cols.issubset(df.columns)
    assert ((0. <= df['order']) & (df['order'] <= 1.)).all()
    assert df['order'].unique().size == np.min([tdata.n_obs, nbins])
    if label is not None:
        lcols = {'label', 'color'}
        assert lcols.issubset(df.columns)
        s_lbl = set(tdata.obs[label])
        assert s_lbl  == set(df['label'])
        assert len(s_lbl) == len(set(df['color']))
        assert f'{label}_colors' in tdata.uns.keys()
