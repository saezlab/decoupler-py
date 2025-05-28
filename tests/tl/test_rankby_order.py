import pandas as pd
import scipy.sparse as sps
import scipy.stats as sts
import pytest

import decoupler as dc


@pytest.mark.parametrize('stat', ['dcor', 'pearsonr', 'spearmanr', 'kendalltau', sts.pearsonr])
def test_rankby_order(
    tdata,
    stat,
):
    df = dc.tl.rankby_order(tdata, order='pstime', stat=stat)
    assert isinstance(df, pd.DataFrame)
    neg_genes = {'G01', 'G02', 'G03', 'G04'}
    pos_genes = {'G05', 'G06', 'G07', 'G08'}
    gt_genes = neg_genes | pos_genes
    pd_genes = set(df.head(len(gt_genes))['name'])
    assert len(gt_genes) > 3
    assert (len(gt_genes & pd_genes) / len(gt_genes)) >= 0.75
    msk = df['name'].isin(gt_genes)
    assert df[~msk]['stat'].mean() < df[msk]['stat'].mean()
    tdata.X = sps.csr_matrix(tdata.X)
    df = dc.tl.rankby_order(tdata, order='pstime', stat=stat)
    assert isinstance(df, pd.DataFrame)
    pd_genes = set(df.head(len(gt_genes))['name'])
    assert len(gt_genes) > 3
    assert (len(gt_genes & pd_genes) / len(gt_genes)) >= 0.75
    msk = df['name'].isin(gt_genes)
    assert df[~msk]['stat'].mean() < df[msk]['stat'].mean()
