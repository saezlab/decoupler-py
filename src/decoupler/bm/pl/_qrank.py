import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from decoupler._docs import docs
from decoupler._Plotter import Plotter
from decoupler.bm.pl._format import _format


@docs.dedent
def qrank(
    df: pd.DataFrame,
    hue: str | None = None,
    palette: str = 'tab20',
    thr_rank: float = 0.5,
    thr_pval: float = 0.05,
    **kwargs
) -> None | Figure:
    """
    Plot 1-qrank and p-value.

    x-axis represent the one minus the quantile normalized ranks for the sources that belong to the ground truth.
    The closer to 1 the better performance is.

    y-axis represents the p-value (-log10) obtained after performing a Ranksums test between the quantile normalized
    ranks of the sources that belong to the ground truth against the sources that do not.
    The higher value the better performance is.

    Parameters
    ----------
    %(df)s
    %(hue)s
    %(palette)s
    thr_rank
        Dashed line to indicate baseline of ranks.
    thr_pval
        Dashed line to indicate baseline of p-values.
    %(plot)s
    """
    # Validate
    assert isinstance(hue, str) or hue is None, 'hue must be str or None'
    assert isinstance(thr_rank, float) and 0. <= thr_rank <= 1., \
    'thr_rank must be float and between 0 and 1'
    assert isinstance(thr_pval, float) and 0. <= thr_pval <= 1., \
    'thr_pval must be float and between 0 and 1'
    # Format
    tmp = _format(df=df, cols=['1-qrank', '-log10(pval)'])
    # Instance
    bp = Plotter(**kwargs)
    # Plot
    if hue is not None:
        sns.scatterplot(
            data=tmp,
            x='1-qrank',
            y='-log10(pval)',
            hue=hue,
            ax=bp.ax,
            palette=palette,
        )
    else:
        sns.scatterplot(
            data=tmp,
            x='1-qrank',
            y='-log10(pval)',
            ax=bp.ax,
        )
    bp.ax.set_xlim(0, 1)
    bp.ax.axvline(x=thr_rank, ls='--', c='black', zorder=0)
    bp.ax.axhline(y=-np.log10(thr_pval), ls='--', c='black', zorder=0)
    bp.ax.set_ylabel(r'$\log_{10}$(pval)')
    if hue is not None:
        bp.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, title=hue)
    return bp._return()
