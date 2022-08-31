import pytest
import numpy as np
import pandas as pd
from anndata import AnnData
from ..decouple import get_wrappers, run_methods, parse_methods, decouple, run_consensus


def test_get_wrappers():
    get_wrappers(['mlm', 'ulm'])


def test_run_methods():
    m = np.array([[7., 1., 1., 1.], [4., 2., 1., 2.], [1., 2., 5., 1.], [1., 1., 6., 2.]])
    r = np.array(['S1', 'S2', 'S3', 'S4'])
    c = np.array(['G1', 'G2', 'G3', 'G4'])
    df = pd.DataFrame(m, index=r, columns=c)
    net = pd.DataFrame([['T1', 'G1', 1], ['T1', 'G2', 2], ['T2', 'G3', -3], ['T2', 'G4', 4]],
                       columns=['source', 'target', 'weight'])
    run_methods(df, net, 'source', 'target', 'weight', ['mlm', 'ulm'], {}, 0, True, False)


def test_parse_methods():
    parse_methods(None, None)
    parse_methods('all', None)
    parse_methods(['mlm', 'ulm'], None)


def test_decouple():
    m = np.array([[7., 1., 1., 1.], [4., 2., 1., 2.], [1., 2., 5., 1.], [1., 1., 6., 2.]])
    r = np.array(['S1', 'S2', 'S3', 'S4'])
    c = np.array(['G1', 'G2', 'G3', 'G4'])
    df = pd.DataFrame(m, index=r, columns=c)
    adata = AnnData(df)
    net = pd.DataFrame([['T1', 'G1', 1], ['T1', 'G2', 2], ['T2', 'G3', -3], ['T2', 'G4', 4]],
                       columns=['source', 'target', 'weight'])
    decouple(adata, net, methods=['mlm', 'ulm'], min_n=0, verbose=True, use_raw=False)
    with pytest.raises(ValueError):
        decouple(adata, net, methods=['mlm', 'ulm', 'asd'], min_n=0, verbose=True, use_raw=False)


def test_run_consensus():
    m = np.array([[7., 1., 1., 1.], [4., 2., 1., 2.], [1., 2., 5., 1.], [1., 1., 6., 2.]])
    r = np.array(['S1', 'S2', 'S3', 'S4'])
    c = np.array(['G1', 'G2', 'G3', 'G4'])
    df = pd.DataFrame(m, index=r, columns=c)
    adata = AnnData(df)
    net = pd.DataFrame([['T1', 'G1', 1], ['T1', 'G2', 2], ['T2', 'G3', -3], ['T2', 'G4', 4]],
                       columns=['source', 'target', 'weight'])
    run_consensus(adata, net, min_n=0, verbose=True, use_raw=False)
