import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pytest

import decoupler as dc

@pytest.fixture
def df():
    df = pd.DataFrame(
        data=[
            ['aucell', 'auroc', 0.45],
            ['aucell', 'auprc', 0.55],
            ['ulm', 'auroc', 0.9],
            ['ulm', 'auprc', 0.8],
            ['aucell', 'recall', 0.45],
            ['aucell', 'precision', 0.55],
            ['ulm', 'recall', 0.9],
            ['ulm', 'precision', 0.8],
            ['aucell', '1-qrank', 0.45],
            ['aucell', '-log10(pval)', 0.9],
            ['ulm', '1-qrank', 0.9],
            ['ulm', '-log10(pval)', 5.6],
        ],
        columns = ['method', 'metric', 'score']
    )
    return df


@pytest.fixture
def hdf(
    df,
):
    hdf = dc.bm.metric.hmean(df)
    return hdf


def test_auc(
    df,
):
    fig = dc.bm.pl.auc(df=df, hue=None, return_fig=True)
    assert isinstance(fig, Figure)
    plt.close(fig)
    fig = dc.bm.pl.auc(df=df, hue='method', return_fig=True)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_fscore(
    df,
):
    fig = dc.bm.pl.fscore(df=df, hue=None, return_fig=True)
    assert isinstance(fig, Figure)
    plt.close(fig)
    fig = dc.bm.pl.fscore(df=df, hue='method', return_fig=True)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_qrank(
    df,
):
    fig = dc.bm.pl.qrank(df=df, hue=None, return_fig=True)
    assert isinstance(fig, Figure)
    plt.close(fig)
    fig = dc.bm.pl.qrank(df=df, hue='method', return_fig=True)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_bar(
    hdf,
):
    fig = dc.bm.pl.bar(df=hdf, x='H(auroc, auprc)', y='method', hue=None, return_fig=True)
    assert isinstance(fig, Figure)
    plt.close(fig)
    fig = dc.bm.pl.bar(df=hdf, x='H(auroc, auprc)', y='method', hue='method', return_fig=True)
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_summary(
    hdf,
):
    fig = dc.bm.pl.summary(df=hdf, y='method', figsize=(6, 3), return_fig=True)
    assert isinstance(fig, Figure)
    plt.close(fig)
