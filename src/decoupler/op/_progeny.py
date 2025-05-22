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
    Pathway RespOnsive GENes for activity inference (PROGENy) :cite:p:`progeny`.

    Wrapper to access PROGENy model gene weights. Each pathway is defined with
    a collection of target genes, each interaction has an associated p-value
    and weight. The top significant interactions per pathway are returned.

    Here is a brief description of each pathway:

    - **Androgen**: involved in the growth and development of the male reproductive organs
    - **EGFR**: regulates growth, survival, migration, apoptosis, proliferation, and differentiation in mammalian cells
    - **Estrogen**: promotes the growth and development of the female reproductive organs
    - **Hypoxia**: promotes angiogenesis and metabolic reprogramming when O2 levels are low
    - **JAK-STAT**: involved in immunity, cell division, cell death, and tumor formation
    - **MAPK**: integrates external signals and promotes cell growth and proliferation
    - **NFkB**: regulates immune response, cytokine production and cell survival
    - **p53**: regulates cell cycle, apoptosis, DNA repair and tumor suppression
    - **PI3K**: promotes growth and proliferation
    - **TGFb**: involved in development, homeostasis, and repair of most tissues
    - **TNFa**: mediates haematopoiesis, immune surveillance, tumour regression and protection from infection
    - **Trail**: induces apoptosis
    - **VEGF**: mediates angiogenesis, vascular permeability, and cell migration
    - **WNT**: regulates organ morphogenesis during development and tissue repair

    Parameters
    ----------
    %(organism)s
    top
        Number of genes per pathway to return. By default all of them.
    thr_padj
        Significance threshold to trim interactions.
    %(license)s
    %(verbose)s

    Returns
    -------
    Dataframe in long format containing target genes for each pathway with their associated weights and p-values.
    """
    # Validate
    assert isinstance(top, (int, float)) and top > 0, \
    'top must be numeric and > 0'
    assert isinstance(thr_padj, (int, float)) and 0. <= thr_padj <= 1., \
    'thr_padj must be numeric and between 0 and 1'
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
    p = p.rename(columns={'pathway': 'source', 'genesymbol': 'target', 'p_value': 'padj'})
    p = p[p['padj'] < thr_padj]
    p = p[['source', 'target', 'weight', 'padj']]
    m = f'progeny - filtered interactions for padj < {thr_padj}'
    _log(m, level='info', verbose=verbose)
    p = p.drop_duplicates(['source', 'target']).reset_index(drop=True)
    return p
