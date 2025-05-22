import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from decoupler._docs import docs
from decoupler._Plotter import Plotter
from decoupler.bm.pl._format import _format


@docs.dedent
def auc(
    df: pd.DataFrame,
    hue: str | None = None,
    palette: str = 'tab20',
    thr_auroc: float = 0.5,
    thr_auprc: float = 0.5,
    **kwargs
) -> None | Figure:
    """
    Plot auroc and auprc.

    x-axis represent the auroc calculated by ranking all obtained enrichment scores, calculating different class thresholds
    and finally obtaining the area under the curve.
    The higher value the better performance is.

    y-axis represent the auprc calculated by ranking all obtained enrichment scores, calculating different class thresholds
    and finally obtaining the area under the curve.
    The higher value the better performance is.

    Parameters
    ----------
    %(df)s
    %(hue)s
    %(palette)s
    thr_auroc
        Dashed line to indicate baseline of auroc.
    thr_auprc
        Dashed line to indicate baseline of auprc.
    %(plot)s
    """
    # Validate
    assert isinstance(hue, str) or hue is None, 'hue must be str or None'
    assert isinstance(thr_auroc, float) and 0. <= thr_auroc <= 1., \
    'thr_auroc must be float and between 0 and 1'
    assert isinstance(thr_auprc, float) and 0. <= thr_auprc <= 1., \
    'thr_auprc must be float and between 0 and 1'
    # Format
    tmp = _format(df=df, cols=['auroc', 'auprc'])
    # Instance
    bp = Plotter(**kwargs)
    # Plot
    if hue is not None:
        sns.scatterplot(
            data=tmp,
            x='auroc',
            y='auprc',
            hue=hue,
            ax=bp.ax,
            palette=palette,
        )
    else:
        sns.scatterplot(
            data=tmp,
            x='auroc',
            y='auprc',
            ax=bp.ax,
        )
    bp.ax.axvline(x=thr_auroc, ls='--', c='black', zorder=0)
    bp.ax.axhline(y=thr_auprc, ls='--', c='black', zorder=0)
    if hue is not None:
        bp.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, title=hue)
    return bp._return()
