import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from decoupler._docs import docs
from decoupler._Plotter import Plotter
from decoupler.bm.pl._format import _format


@docs.dedent
def fscore_scatter(
    df: pd.DataFrame,
    hue: str | None = None,
    palette: str = 'tab20',
    **kwargs
) -> None | Figure:
    """
    Plot precision and recall as scatterplot.

    x-axis represent the recall of correctly predicted sources after filtering by significance.
    The higher value the better performance is.

    x-axis represent the precision of correctly predicted sources after filtering by significance.
    The higher value the better performance is.

    Parameters
    ----------
    %(df)s
    %(hue)s
    %(palette)s
    %(plot)s
    """
    # Validate
    assert isinstance(hue, str) or hue is None, 'hue must be str or None'
    # Format
    tmp = _format(df=df, cols=['recall', 'precision', 'fscore'])
    # Instance
    bp = Plotter(**kwargs)
    # Plot
    args = dict()
    if hue is not None:
        args['hue'] = hue
        args['palette'] = palette
    sns.scatterplot(
        data=tmp,
        x='recall',
        y='precision',
        ax=bp.ax,
        **args
    )
    if hue is not None:
        bp.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, title=hue)
    return bp._return()


@docs.dedent
def fscore_bar(
    df: pd.DataFrame,
    y: str,
    hue: str | None = None,
    palette: str = 'tab20',
    **kwargs
) -> None | Figure:
    """
    Plot precision and recall as barplot.

    x-axis represent the f-score, which is the harmonic mean between precision and recall of
    sources after filtering by significance.

    y-axis represent a grouping variable.

    Parameters
    ----------
    %(df)s
    y
        Grouping variable to plot on y axis.
    %(hue)s
    %(palette)s
    %(plot)s
    """
    # Validate
    assert isinstance(y, str), 'y must be str'
    assert isinstance(hue, str) or hue is None, 'hue must be str or None'
    # Format
    tmp = _format(df=df, cols=['recall', 'precision', 'fscore'])
    # Instance
    bp = Plotter(**kwargs)
    # Plot
    order = (
        tmp
        .groupby(y)['fscore']
        .mean()
        .sort_values(ascending=False)
        .index
    )
    args = dict()
    if hue is not None:
        args['hue'] = hue
        args['palette'] = palette
    sns.barplot(
        data=tmp,
        y=y,
        x='fscore',
        order=order,
        **args
    )
    if hue is not None and hue != y:
        bp.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, title=hue)
    return bp._return()
