import numpy as np
import pandas as pd

from decoupler._docs import docs
from decoupler._log import _log
from decoupler.op._resource import resource


@docs.dedent
def progeny(
    organism: str = 'human',
    top: int | float = np.inf,
    thr_padj: float = 0.05,
    license: str = 'academic',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Pathway RespOnsive GENes for activity inference (PROGENy).

    Wrapper to access PROGENy model gene weights. Each pathway is defined with
    a collection of target genes, each interaction has an associated p-value
    and weight. The top significant interactions per pathway are returned.

    Parameters
    ----------
    %(organism)s
    top
        Number of genes per pathway to return. By default all of them.
    thr_padj
        Significance threshold to trim interactions.
    %(license)s

    Returns
    -------
    Dataframe in long format containing target genes for each pathway with their associated weights and p-values.
    """
    # Validate
    assert isinstance(top, (int, float)) and top > 0, 'top must be numeric and > 0'
    assert isinstance(thr_padj, float) and 1 >= thr_padj >= 0, 'thr_padj must be numeric and between 0 and 1'
    # Download
    p = resource(name='PROGENy', organism=organism, license=license, verbose=verbose)
    p = (
        p
        .sort_values('p_value')
        .groupby('pathway')
        .head(top)
        .sort_values(['pathway', 'p_value'])
        .reset_index(drop=True)
    )
    p = p.rename(columns={'pathway': 'source', 'genesymbol': 'target', 'p_value': 'pval'})
    p = p[p['pval'] < thr_padj]
    p = p[['source', 'target', 'weight', 'pval']]
    m = f'progeny - filtered interactions for padj < {thr_padj}'
    _log(m, level='info', verbose=verbose)
    p = p.drop_duplicates(['source', 'target']).reset_index(drop=True)
    return p
