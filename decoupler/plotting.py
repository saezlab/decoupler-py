import numpy as np
import pandas as pd

from .pre import extract, rename_net


def check_if_matplotlib(return_mpl=False):
    if not return_mpl:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            raise ImportError('matplotlib is not installed. Please install it with: pip install matplotlib')
        return plt
    else:
        try:
            import matplotlib as mpl
        except Exception:
            raise ImportError('matplotlib is not installed. Please install it with: pip install matplotlib')
        return mpl


def check_if_seaborn():
    try:
        import seaborn as sns
    except Exception:
        raise ImportError('seaborn is not installed. Please install it with: pip install seaborn')
    return sns


def save_plot(fig, ax, save):
    if save is not None:
        if ax is not None:
            fig.savefig(save)
        else:
            raise ValueError("ax is not None, cannot save figure.")


def set_limits(vmin, vcenter, vmax, values):

    if vmin is None:
        vmin = values.min()

    if vmax is None:
        vmax = values.max()

    if vcenter is None:
        vcenter = values.mean()

    return vmin, vcenter, vmax


def plot_volcano(logFCs, pvals, contrast, name=None, net=None, top=5, source='source', target='target',
                 weight='weight', sign_thr=0.05, lFCs_thr=0.5, figsize=(7, 5), dpi=100, ax=None,
                 return_fig=False, save=None):
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
        Whether to return a Figure object or not.
    save : str, None
        Path to where to save the plot. Infer the filetype if ending on {`.pdf`, `.png`, `.svg`}.
    """

    # Load plotting packages
    plt = check_if_matplotlib()

    # Transform sign_thr
    sign_thr = -np.log10(sign_thr)

    # Plot
    fig = None
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
    signs = df[(np.abs(df['logFCs']) >= lFCs_thr) & (df['pvals'] >= sign_thr)].sort_values('pvals', ascending=False)
    signs = signs.iloc[:top]

    # Add labels
    ax.set_ylabel('-log10(pvals)')
    texts = []
    for x, y, s in zip(signs['logFCs'], signs['pvals'], signs.index):
        texts.append(ax.text(x, y, s))

    save_plot(fig, ax, save)

    if return_fig and ax is None:
        return fig


def plot_violins(mat, thr=None, log=False, use_raw=False, figsize=(7, 5), dpi=100, ax=None, title=None, ylabel=None,
                 color='#1F77B4', return_fig=False, save=None):
    """
    Plot distribution of features' values per sample.

    Parameters
    ----------
    mat : list, DataFrame or AnnData
        List of [features, matrix], dataframe (samples x features) or an AnnData instance.
    thr : float
        Threshold to plot horizontal line.
    log : bool
        Whether to log1p the data or not.
    use_raw : bool
        Use raw attribute of mat if present.
    figsize : tuple
        Figure size.
    dpi : int
        DPI resolution of figure.
    ax : Axes, None
        A matplotlib axes object. If None returns new figure.
    title : str
        Text to write as title of the plot.
    ylabel : str
        Text to write as ylabel of the plot.
    color : str
        Color to plot the violins.
    return_fig : bool
        Whether to return a Figure object or not.
    save : str, None
        Path to where to save the plot. Infer the filetype if ending on {`.pdf`, `.png`, `.svg`}.
    """

    # Load plotting packages
    sns = check_if_seaborn()
    plt = check_if_matplotlib()

    # Extract data
    m, r, c = extract(mat, use_raw=use_raw)

    # Format
    x = np.repeat(r, m.shape[1])
    y = m.A.flatten()

    # Plot
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # log(x+1) transform
    if log:
        y = np.log1p(y)

    sns.violinplot(x=x, y=y, ax=ax, color=color)
    if thr is not None:
        ax.axhline(y=thr, linestyle='--', color="black")
    ax.tick_params(axis='x', rotation=90)

    # Label
    if title is not None:
        ax.set_title(title)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    save_plot(fig, ax, save)

    if return_fig and ax is None:
        return fig


def plot_barplot(acts, contrast, top=25, vertical=False, cmap='coolwarm', vmin=None, vcenter=0, vmax=None,
                 figsize=(7, 5), dpi=100, return_fig=False, save=None):
    """
    Plot barplots showing the top absolute value activities.

    Parameters
    ----------
    acts : DataFrame
        Activities obtained from any method.
    contrast : str
        Name of the contrast (row) to plot.
    top : int
        Number of top features to plot.
    vertical : bool
        Whether to plot verticaly or horizontaly.
    cmap : str
        Colormap to use.
    vmin : float, None
        The value representing the lower limit of the color scale.
    vcenter : float, None
        The value representing the center of the color scale.
    vmax : float, None
        The value representing the upper limit of the color scale.
    figsize : tuple
        Figure size.
    dpi : int
        DPI resolution of figure.
    return_fig : bool
        Whether to return a Figure object or not.
    save : str, None
        Path to where to save the plot. Infer the filetype if ending on {`.pdf`, `.png`, `.svg`}.
    """
    # Load plotting packages
    sns = check_if_seaborn()
    plt = check_if_matplotlib()
    mpl = check_if_matplotlib(return_mpl=True)

    # Process df
    df = acts.loc[[contrast]]

    df = (df

          # Sort by absolute value and transpose
          .iloc[:, np.argsort(abs(df.values))[0]].T

          # Select top features and add index col
          .tail(top).reset_index()

          # Rename col
          .rename({contrast: 'acts'}, axis=1)

          # Sort by activities
          .sort_values('acts'))

    if vertical:
        x, y = 'acts', 'index'
    else:
        x, y = 'index', 'acts'

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    sns.barplot(data=df, x=x, y=y, ax=ax)

    if vertical:
        sizes = np.array([bar.get_width() for bar in ax.containers[0]])
        ax.set_xlabel('Activity')
        ax.set_ylabel('')
    else:
        sizes = np.array([bar.get_height() for bar in ax.containers[0]])
        ax.tick_params(axis='x', rotation=90)
        ax.set_ylabel('Activity')
        ax.set_xlabel('')

    # Compute color limits
    vmin, vcenter, vmax = set_limits(vmin, vcenter, vmax, df['acts'])

    # Rescale cmap
    divnorm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    cmap_f = plt.get_cmap(cmap)
    div_colors = cmap_f(divnorm(sizes))
    for bar, color in zip(ax.containers[0], div_colors):
        bar.set_facecolor(color)

    # Add legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=divnorm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax)

    save_plot(fig, ax, save)

    if return_fig and ax is None:
        return fig
