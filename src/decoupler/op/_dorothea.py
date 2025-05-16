import numpy as np
import pandas as pd

from decoupler._docs import docs
from decoupler._log import _log
from decoupler._download import URL_INT, _download
from decoupler.op._translate import translate
from decoupler.op._dtype import _infer_dtypes


@docs.dedent
def dorothea(
    organism: str = 'human',
    levels: str | list = ['A', 'B', 'C'],
    dict_weights: dict | None = None,
    license: str = 'academic',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    DoRothEA gene regulatory network :cite:p:`dorothea`.

    Wrapper to access DoRothEA gene regulatory network. DoRothEA is a
    comprehensive resource containing a curated collection of transcription
    factors (TFs) and their target genes. Each interaction is weighted by its
    mode of regulation (either positive or negative) and by its confidence
    level.

    Parameters
    ----------
    %(organism)s
    levels
        List of confidence levels to return. Goes from A to D, A being the
        most confident and D being the less.
    dict_weights
        Dictionary of values to divide the mode of regulation (-1 or 1),
        one for each confidence level. Bigger values will generate weights
        close to zero.
    %(license)s
    %(verbose)s

    Returns
    -------
    Dataframe in long format containing target genes for each TF with their associated weights and confidence level.
    """
    assert isinstance(levels, (str, list)), 'levels must be str or list'
    if isinstance(levels, str):
        levels = [levels]
    assert all(l in {'A', 'B', 'C', 'D'} for l in levels), 'levels can only contain any of these values: A, B, C, and/or D'
    assert isinstance(dict_weights, dict) or dict_weights is None, 'dict_weights must be dict or None'
    if dict_weights:
        assert all(k in levels for k in dict_weights), f'dict_weights keys must be in levels={levels}'
        weights = dict_weights
    else:
        weights = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
        weights = {k: weights[k] for k in weights if k in levels}
    # Read
    str_levels = ','.join(levels)
    url_ext = f'datasets=dorothea&dorothea_levels={str_levels}&fields=dorothea_level&license={license}'
    url = URL_INT + url_ext
    m = f'dorothea - Accessing DoRothEA (levels {str_levels}) with {license} license and weights={weights}'
    _log(m, level='info', verbose=verbose)
    do = _download(url, sep='\t', verbose=verbose)
    # Filter extra columns
    do = do[[
        'source_genesymbol', 'target_genesymbol',
        'is_stimulation', 'is_inhibition',
        'consensus_direction', 'consensus_stimulation',
        'consensus_inhibition', 'dorothea_level',
    ]]
    # Remove duplicates
    do = do[~do.duplicated(['source_genesymbol', 'dorothea_level', 'target_genesymbol'])]
    # Assign top level if more than 2
    do['dorothea_level'] = [lvl.split(';')[0] for lvl in do['dorothea_level']]
    # Assign mode of regulation
    mor = []
    for i in do.itertuples():
        if i.is_stimulation and i.is_inhibition:
            if i.consensus_stimulation:
                mor.append(1)
            else:
                mor.append(-1)
        elif i.is_stimulation:
            mor.append(1)
        elif i.is_inhibition:
            mor.append(-1)
        else:
            mor.append(1)
    do['mor'] = mor
    # Compute weight based on confidence: mor/confidence
    do['weight'] = [i.mor / weights[i.dorothea_level] for i in do.itertuples()]
    # Format
    do = (
        do
        .rename(columns={'source_genesymbol': 'source', 'target_genesymbol': 'target', 'dorothea_level': 'confidence'})
        [['source', 'target', 'weight', 'confidence']]
        .sort_values('confidence')
    )
    do = do[do['confidence'].isin(levels)].reset_index(drop=True)
    do = _infer_dtypes(do)
    if organism != 'human':
        do = translate(do, columns=['source', 'target'], target_organism=organism, verbose=verbose)
    do = do.drop_duplicates(['source', 'target']).reset_index(drop=True)
    return do
