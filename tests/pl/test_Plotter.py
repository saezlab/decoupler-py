import tempfile

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pytest

import decoupler as dc


def test_plot_ax(
    adata
):
    fig, axes = plt.subplots(1, 2, tight_layout=True, figsize=(4, 2))
    ax1, ax2 = axes
    dc.pl.obsbar(adata=adata, y='group', hue='sample', ax=ax1)
    dc.pl.obsbar(adata=adata, y='sample', hue='group', ax=ax2)


def test_plot_save(
    adata
):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
        fig = dc.pl.obsbar(adata=adata, y='group', hue='sample', save=tmp.name)
        tmp.flush()
        img = mpimg.imread(tmp.name)
        assert img is not None
