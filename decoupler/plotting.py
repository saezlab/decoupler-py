import numpy as np
import pandas as pd

from .pre import rename_net


def check_if_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except Exception:
        raise ImportError('matplotlib is not installed. Please install it with: pip install matplotlib')
    return plt


def plot_volcano(logFCs, pvals, contrast, name=None, net=None, top=5, source='source', target='target', weight='weight', sign_thr=0.05,
                 lFCs_thr=0.5, figsize=(7, 5), dpi=100, ax=None, return_fig=False, save=None):
    """
    Plot logFC and p-values. If name and net are provided, it does the same for the targets of a selected source.

    Parameters
    ----------
    logFCs : DataFrame
        Data-frame of logFCs (contrasts x features).
    pvals : DataFrame
        Data-frame of p-values (contrasts x features).
    contrast : str
        Name of the contrast (row) to plot.
    name : str, None
        Name of the source to plot. If None, plot classic volcano (without subsetting targets).
    net : DataFrame, None
        Network dataframe. If None, plot classic volcano (without subsetting targets).
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
    ax : Axes, None
        A matplotlib axes object. If None returns new figure.
    return_fig : bool
        Wether to return a Figure object or not.
    save : str, None
        Path to where to save the plot. Infer the filetype if ending on {`.pdf`, `.png`, `.svg`}.
    """

    plt = check_if_matplotlib()
    sign_thr = -np.log10(sign_thr)
    
    # Plot
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    
    # Check for net
    if net is not None:
        # Rename nets
        net = rename_net(net, source=source, target=target, weight=weight)

        # Get max and if + and -
        max_n = np.std(np.abs(net['weight']), ddof=1)*2
        has_neg = np.any(net['weight'] < 0)

        # Filter by shared targets
        if name is None:
            raise ValueError('If net is given, name cannot be None.')
        df = net[net[source] == name].set_index('target')
        df['logFCs'] = logFCs.loc[[contrast]].T
        df['pvals'] = -np.log10(pvals.loc[[contrast]].T)
        df = df[~np.any(pd.isnull(df), axis=1)]
        
        if has_neg:
            vmin = -max_n
        else:
            vmin = 0
        df.plot.scatter(x='logFCs', y='pvals', c='weight', cmap='coolwarm',
                                vmin=vmin, vmax=max_n, sharex=False, ax=ax)
        ax.set_title('{0} | {1}'.format(contrast, name))
    else:
        df = logFCs.loc[[contrast]].T.rename({contrast: 'logFCs'}, axis=1)
        df['pvals'] = -np.log10(pvals.loc[[contrast]].T)
        df = df[~np.any(pd.isnull(df), axis=1)]
        df['weight'] = 'gray'
        df.loc[(df['logFCs'] >= lFCs_thr) & (df['pvals'] >= sign_thr), 'weight'] = '#D62728'
        df.loc[(df['logFCs'] <= -lFCs_thr) & (df['pvals'] >= sign_thr), 'weight'] = '#1F77B4'
        df.plot.scatter(x='logFCs', y='pvals', c='weight', cmap='coolwarm', sharex=False, ax=ax)
        ax.set_title('{0}'.format(contrast))
    
    # Draw sign lines
    ax.axhline(y=sign_thr, linestyle='--', color="black")
    ax.axvline(x=lFCs_thr, linestyle='--', color="black")
    ax.axvline(x=-lFCs_thr, linestyle='--', color="black")
    
    # Plot top sign features
    signs = df[(np.abs(df['logFCs']) >= lFCs_thr) &
                       (df['pvals'] >= sign_thr)].sort_values('pvals', ascending=False)
    signs = signs.iloc[:top]
    
    # Add labels
    ax.set_ylabel('-log10(pvals)')
    texts = []
    for x, y, s in zip(signs['logFCs'], signs['pvals'], signs.index):
        texts.append(ax.text(x, y, s))

    if save is not None:
        if ax is not None:
            fig.savefig(save)
        else:
            raise ValueError("ax is not None, cannot save figure.")

    if return_fig:
        return fig