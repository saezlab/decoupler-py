import numpy as np
import pandas as pd

from decoupler._docs import docs
from decoupler._log import _log
from decoupler._download import URL_INT, _download
from decoupler.op._translate import translate
from decoupler.op._dtype import _infer_dtypes


@docs.dedent
def collectri(
    organism: str = 'human',
    remove_complexes: bool = False,
    license: str = 'academic',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    CollecTRI gene regulatory network :cite:p:`collectri`.

    Wrapper to access CollecTRI gene regulatory network. CollecTRI is a
    comprehensive resource containing a curated collection of transcription
    factors (TFs) and their target genes. It is an expansion of DoRothEA.
    Each interaction is weighted by its mode of regulation (either positive or negative).

    Parameters
    ----------
    %(organism)s
    remove_complexes
        Whether to remove complexes.
    %(license)s
    %(verbose)s

    Returns
    -------
    Dataframe in long format containing target genes for each TF with their associated weights,
    and if available, the PMIDs supporting each interaction.
    """
    url = 'https://zenodo.org/records/8192729/files/CollecTRI_regulons.csv?download=1'
    ct = _download(url, verbose=verbose)
    # Update resources
    resources = []
    for str_res in ct['resources']:
        lst_res = str_res.replace('CollecTRI', '').split(';')
        str_res = ';'.join(sorted([res.replace('_', '') for res in lst_res if res != '']))
        resources.append(str_res)
    ct['resources'] = resources
    # Format references
    ct['references'] = ct['references'].str.replace('CollecTRI:', '')
    ct = ct.dropna()
    if remove_complexes:
        ct = ct[~ct['source'].isin(['AP1', 'NFKB'])]
    ct = _infer_dtypes(ct)
    if organism != 'human':
        ct = translate(ct, columns=['source', 'target'], target_organism=organism, verbose=verbose)
    ct = ct.drop_duplicates(['source', 'target']).reset_index(drop=True)
    return ct
