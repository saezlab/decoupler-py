import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from decoupler._docs import docs
from decoupler._Plotter import Plotter
from decoupler.bm.pl._format import _format


@docs.dedent
def bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    palette: str = 'tab20',
    **kwargs
) -> None | Figure:
    """
    Plot the harmonic mean between two metric statistics as a barplot.

    x-axis represent the harmonic mean between metric statistics.

    y-axis represent a grouping variable.

    Parameters
    ----------
    %(df)s
    x
        Continous variable to plot on x axis.
    %(y)s
    %(hue)s
    %(palette)s
    %(plot)s
    """
    # Validate
    assert isinstance(x, str), 'x must be str'
    assert isinstance(y, str), 'y must be str'
    assert isinstance(hue, str) or hue is None, 'hue must be str or None'
    # Instance
    bp = Plotter(**kwargs)
    # Plot
    order = (
        df
        .groupby(y)[x]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    args = dict()
    if hue is not None:
        args['hue'] = hue
        args['palette'] = palette
    sns.barplot(
        data=df,
        y=y,
        x=x,
        order=order,
        **args
    )
    if hue is not None and hue != y:
        bp.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, title=hue)
    return bp._return()
