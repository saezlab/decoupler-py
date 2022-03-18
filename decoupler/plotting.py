import numpy as np
import pandas as pd

from .pre import rename_net


def check_if_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except Exception:
        raise BaseException('omnipath is not installed. Please install it with: pip install omnipath')
    return plt


def plot_volcano(logFCs, pvals, name, contrast, net, top=5, source='source', target='target', weight='weight', sign_thr=0.05,
                 lFCs_thr=0.5, figsize=(7, 5), dpi=100):
    """
    Plot logFC and p-values of a selected source by a specific contrast.

    Parameters
    ----------
    logFCs : DataFrame
        Data-frame of logFCs (contrasts x features).
    pvals : DataFrame
        Data-frame of p-values (contrasts x features).
    name : str
        Name of the source to plot.
    contrast : str
        Name of the contrast (row) to plot.
    net : DataFrame
        Network dataframe.
    top : int
        Number of top differentially expressed features.
    source : str
        Column name in net with source nodes.
    target : str
        Column name in net with target nodes.
    weight : str
        Column name in net with weights.
    sign_thr : float
        Significance threshold for p-values.
    lFCs_thr : float
        Significance threshold for logFCs.
    figsize : tuple
        Figure size.
    dpi : int
        DPI resolution of figure.
    """

    plt = check_if_matplotlib()

    # Rename nets
    net = rename_net(net, source=source, target=target, weight=weight)

    # Get max and if + and -
    max_n = np.std(np.abs(net['weight']), ddof=1)*2
    has_neg = np.any(net['weight'] < 0)

    # Filter by shared targets
    source_net = net[net[source] == name].set_index('target')
    source_net['logFC'] = logFCs.loc[[contrast]].T
    source_net['pvals'] = -np.log10(pvals.loc[[contrast]].T)
    source_net = source_net[~np.any(pd.isnull(source_net), axis=1)]

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    if has_neg:
        source_net.plot.scatter(x='logFC', y='pvals', c='weight', cmap='coolwarm',
                                vmin=-max_n, vmax=max_n, sharex=False, ax=ax)
    else:
        source_net.plot.scatter(x='logFC', y='pvals', c='weight', cmap='coolwarm',
                                vmin=0, vmax=max_n, sharex=False, ax=ax)
    signs = source_net[(np.abs(source_net['logFC']) > lFCs_thr) &
                       (source_net['pvals'] > -np.log10(0.05))].sort_values('pvals', ascending=False)
    signs = signs.iloc[:top]
    ax.axhline(y=-np.log10(sign_thr), linestyle='--', color="black")
    ax.axvline(x=lFCs_thr, linestyle='--', color="black")
    ax.axvline(x=-lFCs_thr, linestyle='--', color="black")
    ax.set_ylabel('-log10(pvals)')
    ax.set_title('{0} | {1}'.format(contrast, name))

    # Add labels
    texts = []
    for x, y, s in zip(signs['logFC'], signs['pvals'], signs.index):
        texts.append(ax.text(x, y, s))
