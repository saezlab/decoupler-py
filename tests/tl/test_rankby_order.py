import pandas as pd
import scipy.sparse as sps

import decoupler as dc


def test_rankby_order():
    adata, net = dc.ds.toy(pstime=True)
    df = dc.tl.rankby_order(adata, order='pstime')
    assert isinstance(df, pd.DataFrame)
    neg_genes = {'G01', 'G02', 'G03', 'G04'}
    pos_genes = {'G05', 'G06', 'G07', 'G08'}
    gt_genes = neg_genes | pos_genes
    pd_genes = set(df[df['padj'] < 0.05]['name'])
    assert gt_genes == pd_genes
    msk = df['name'].isin(gt_genes)
    assert df[~msk]['impr'].mean() < df[msk]['impr'].mean()
    assert (df[df['name'].isin(neg_genes)]['sign'] == -1).all()
    assert (df[df['name'].isin(pos_genes)]['sign'] == +1).all()
    adata.X = sps.csr_matrix(adata.X)
    df = dc.tl.rankby_order(adata, order='pstime')
    assert isinstance(df, pd.DataFrame)
    pd_genes = set(df[df['padj'] < 0.05]['name'])
    assert gt_genes == pd_genes
    msk = df['name'].isin(gt_genes)
    assert df[~msk]['impr'].mean() < df[msk]['impr'].mean()
    assert (df[df['name'].isin(neg_genes)]['sign'] == -1).all()
    assert (df[df['name'].isin(pos_genes)]['sign'] == +1).all()
