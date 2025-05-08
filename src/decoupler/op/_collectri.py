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
    split_complexes: bool = False,
    license: str = 'academic',
    verbose: bool = False,
) -> pd.DataFrame:
    """
    CollecTRI gene regulatory network.

    Wrapper to access CollecTRI gene regulatory network. CollecTRI is a
    comprehensive resource containing a curated collection of transcription
    factors (TFs) and their target genes. It is an expansion of DoRothEA.
    Each interaction is weighted by its mode of regulation (either positive or negative).

    Parameters
    ----------
    %(organism)s
    split_complexes
        Whether to split complexes into subunits. By default complexes are kept as they are.
    %(license)s

    Returns
    -------
    Dataframe in long format containing target genes for each TF with their associated weights,
    and if available, the PMIDs supporting each interaction.
    """
    # Validate
    assert isinstance(split_complexes, bool), 'split_complexes must be bool'
    # Download
    url_ext = f'datasets=collectri&fields=references&license={license}'
    url = URL_INT + url_ext
    m = f'collectri - Accessing CollecTRI with {license} license'
    _log(m, level='info', verbose=verbose)
    ct = _download(url, sep='\t', verbose=verbose)
    ct = ct[['source_genesymbol', 'target_genesymbol', 'is_inhibition', 'references']]
    ct.loc[:, 'pmid'] = (
        ct
        .loc[~ct['references'].isna(), "references"]
        .str.findall(r":(\d+)").apply(lambda x: ";".join(sorted(set(x))))
    )
    ct['weight'] = np.where(ct['is_inhibition'], -1, 1)
    ct = ct.rename(columns={'source_genesymbol': 'source', 'target_genesymbol': 'target'})[['source', 'target', 'weight', 'pmid']]
    ct = _infer_dtypes(ct)
    if organism != 'human':
        ct = translate(ct, columns=['source', 'target'], target_organism=organism, verbose=verbose)
    if not split_complexes:
        updated_names = []
        for gene in ct['source']:
            ugene = gene.upper()
            if ugene.startswith('JUN') or ugene.startswith('FOS'):
                updated_names.append('AP1')
            elif ugene.startswith('REL') or ugene.startswith('NFKB'):
                updated_names.append('NFKB')
            else:
                updated_names.append(gene)
        ct.loc[:, 'source'] = updated_names
    ct = ct.drop_duplicates(['source', 'target']).reset_index(drop=True)
    return ct
