import pandas as pd
import pytest

import decoupler as dc


@pytest.mark.parametrize(
    'groupby,reference,method',
    [
        ['group', 'rest', 'wilcoxon'],
        ['group', 'A', 't-test'],
        ['group', 'A', 't-test_overestim_var'],
        ['sample', 'rest', 't-test_overestim_var'],
        ['sample', 'S01', 't-test_overestim_var'],
        ['sample', ['S01'], 't-test_overestim_var'],
        ['sample', ['S01', 'S02'], 't-test_overestim_var'],
    ]
)
def test_rankby_group(
    adata,
    groupby,
    reference,
    method,
):
    df = dc.tl.rankby_group(adata=adata, groupby=groupby, reference=reference, method=method)
    assert isinstance(df, pd.DataFrame)
    cols_cat = {'group', 'reference', 'name'}
    cols_num = {'stat', 'meanchange', 'pval', 'padj'}
    cols = cols_cat | cols_num
    assert cols.issubset(set(df.columns))
    for col in cols_cat:
        assert isinstance(df[col].dtype, pd.CategoricalDtype)
    for col in cols_num:
        assert pd.api.types.is_numeric_dtype(df[col])
    assert set(df['group'].cat.categories).issubset(set(adata.obs[groupby].cat.categories))
    assert ((0. <= df['padj']) & (df['padj'] <= 1.)).all()
