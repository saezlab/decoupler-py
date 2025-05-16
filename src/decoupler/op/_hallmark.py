import numpy as np
import pandas as pd

from decoupler._docs import docs
from decoupler._log import _log
from decoupler._download import URL_INT, _download
from decoupler.op._translate import translate
from decoupler.op._dtype import _infer_dtypes


@docs.dedent
def hallmark(
    organism: str = 'human',
    license: str = 'academic',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Hallmark gene sets :cite:p:`msigdb`.

    Hallmark gene sets summarize and represent specific well-defined
    biological states or processes and display coherent expression.

    Parameters
    ----------
    %(organism)s
    %(license)s
    %(verbose)s

    Returns
    -------
    Dataframe in long format containing the hallmark gene sets.
    """
    url = 'https://static.omnipathdb.org/tables/msigdb-hallmark.tsv.gz'
    hm = _download(url, sep='\t', compression='gzip', verbose=verbose)
    hm = hm[['geneset', 'genesymbol']]
    hm = hm.rename(columns={'geneset': 'source', 'genesymbol': 'target'})
    hm['source'] = hm['source'].str.replace('HALLMARK_', '')
    hm['target'] = hm['target'].str.replace('COMPLEX:', '')
    hm = hm.explode('target')
    hm = _infer_dtypes(hm)
    if organism != 'human':
        hm = translate(hm, columns=['source', 'target'], target_organism=organism, verbose=verbose)
    hm = hm.drop_duplicates(['source', 'target']).reset_index(drop=True)
    return hm
