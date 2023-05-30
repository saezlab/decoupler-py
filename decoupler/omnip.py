"""
Utility functions to query OmniPath.
Functions to retrieve resources from the meta-database OmniPath.
"""

from __future__ import annotations

__all__ = [
    'get_progeny',
    'show_resources',
    'get_resource',
    'get_dorothea',
    'translate_net',
]

import os
import builtins
from types import ModuleType
from typing import Iterable
from typing_extensions import Literal
import warnings
import numpy as np
import pandas as pd

builtins.PYPATH_LOG = os.devnull
PYPATH_MIN_VERSION = '0.14.28'
ORGANISMS = {
    'human': (
        'human', 'h. sapiens', 'hsapiens',
        '9606', 9606, 'homo sapiens',
    ),
    'mouse': (
        'mouse', 'm. musculus', 'mmusculus',
        '10090', 10090, 'mus musculus',
    ),
    'rat': (
        'rat', 'r. norvegicus', 'rnorvegicus',
        '10116', 10116, 'rattus norvegicus',
    ),
    'fly': (
        'fly', 'd. melanogaster', 'dmelanogaster',
        '7227', 7227, 'drosphila melanogaster',
    ),
}
DOROTHEA_LEVELS = Literal['A', 'B', 'C', 'D']


def _check_if_omnipath() -> ModuleType:

    try:

        import omnipath as op

    except Exception:

        raise ImportError(
            'omnipath is not installed. Please install it by: '
            '`pip install omnipath`.'
        )

    return op


def _is_organism(
        name: str | int,
        organism: Literal['human', 'mouse', 'rat'],
        ) -> bool:
    """
    Tells if `name` means one of human, mouse, rat.
    """

    return str(name).lower() in ORGANISMS.get(organism.lower(), ())


def _is_mouse(name: str) -> bool:
    """
    Does the organism name or ID mean mouse?
    """

    return _is_organism(name, 'mouse')


def _is_human(name: str) -> bool:
    """
    Does the organism name or ID mean human?
    """

    return _is_organism(name, 'human')


def _is_rat(name: str) -> bool:
    """
    Does the organism name or ID mean human?
    """

    return _is_organism(name, 'rat')


def get_progeny(organism: str | int = 'human', top: int = 100) -> pd.DataFrame:
    """
    Pathway RespOnsive GENes for activity inference (PROGENy).

    Wrapper to access PROGENy model gene weights. Each pathway is defined with
    a collection of target genes, each interaction has an associated p-value
    and weight. The top significant interactions per pathway are returned.

    Parameters
    ----------
    organism : str
        The organism of interest: either NCBI Taxonomy ID, common name,
        latin name or Ensembl name. Organisms other than human will be
        translated from human data by orthology.
    top : int
        Number of genes per pathway to return.

    Returns
    -------
    p : DataFrame
        Dataframe in long format containing target genes for each pathway with
        their associated weights and p-values.
    """

    op = _check_if_omnipath()

    p = op.requests.Annotations.get(resources='PROGENy')
    p = p.set_index([
        'record_id', 'uniprot', 'genesymbol',
        'entity_type', 'source', 'label',
    ])
    p = p.unstack('label').droplevel(axis=1, level=0)
    p.columns = np.array(p.columns)
    p = p.reset_index()
    p.columns.name = None
    p = p[['genesymbol', 'p_value', 'pathway', 'weight']]
    p = p[~p.duplicated(['pathway', 'genesymbol'])]
    p['p_value'] = p['p_value'].astype(np.float32)
    p['weight'] = p['weight'].astype(np.float32)
    p = (
        p.
        sort_values('p_value').
        groupby('pathway').
        head(top).
        sort_values(['pathway', 'p_value']).
        reset_index()
    )
    p = p[['pathway', 'genesymbol', 'weight', 'p_value']]
    p['weight'] = p['weight'].astype(np.float32)
    p['p_value'] = p['p_value'].astype(np.float32)
    p.columns = ['source', 'target', 'weight', 'p_value']

    if not _is_human(organism):

        p = translate_net(
            p,
            columns='target',
            source_organism=9606,
            target_organism=organism,
        )

    return p.reset_index(drop=True)


def get_resource(name: str, organism: str | int = 'human') -> pd.DataFrame:
    """
    Wrapper to access resources inside Omnipath.

    This wrapper allows to easly query different prior knowledge resources. To
    check available resources run ``decoupler.show_resources()``. For more
    information visit the official website for
    [Omnipath](https://omnipathdb.org/).

    Parameters
    ----------
    name : str
        Name of the resource to query.
    organism : int | str
        The organism of interest: either NCBI Taxonomy ID, common name,
        latin name or Ensembl name. Organisms other than human will be
        translated from human data by orthology.

    Returns
    -------
    df : DataFrame
        Dataframe in long format relating genes to biological entities.
    """

    resources = show_resources()
    msg = (
        f'{name} is not a valid resource. Please, run '
        'decoupler.show_resources to see the list of available resources.'
    )
    assert name in resources, msg

    op = _check_if_omnipath()

    df = op.requests.Annotations.get(resources=name, entity_type='protein')
    df = df.set_index([
        'record_id', 'uniprot',
        'genesymbol', 'entity_type',
        'source', 'label',
    ])
    df = df.unstack('label').droplevel(axis=1, level=0)
    df = df.drop(
        columns=[name for name in df.index.names if name in df.columns]
    )
    df.columns = list(df.columns)
    df = df.reset_index()
    df = df.drop(columns=['record_id', 'uniprot', 'entity_type', 'source'])

    if not _is_human(organism):

        df = translate_net(
            df,
            target_organism=organism,
            columns='genesymbol',
            unique_by=None,
        )

    return df.reset_index(drop=True)


def show_resources() -> list:
    """
    Shows available resources in Omnipath. For more information visit the
    official website for [Omnipath](https://omnipathdb.org/).

    Returns
    -------
    lst : list
        List of available resources to query with `dc.get_resource`.
    """

    op = _check_if_omnipath()

    return list(op.requests.Annotations.resources())


def get_dorothea(
        organism: str | int = 'human',
        levels: DOROTHEA_LEVELS | Iterable[DOROTHEA_LEVELS] = ('A', 'B', 'C'),
        weight_dict: dict[str, int] | None = None,
        ) -> pd.DataFrame:
    """
    DoRothEA gene regulatory network.

    Wrapper to access DoRothEA gene regulatory network. DoRothEA is a
    comprehensive resource containing a curated collection of transcription
    factors (TFs) and their target genes. Each interaction is weighted by its
    mode of regulation (either positive or negative) and by its confidence
    level.

    Parameters
    ----------
    organism : str
        The organism of interest: either NCBI Taxonomy ID, common name,
        latin name or Ensembl name. Organisms other than human will be
        translated from human data by orthology.
    levels : list
        List of confidence levels to return. Goes from A to D, A being the
        most confident and D being the less.
    weight_dict : dict
        Dictionary of values to divide the mode of regulation (-1 or 1),
        one for each confidence level. Bigger values will generate weights
        close to zero.

    Returns
    -------
    do : DataFrame
        Dataframe in long format containing target genes for each TF with
        their associated weights and confidence level.
    """

    levels = list(levels)
    weights = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
    weights.update(weight_dict or {})

    _organism = (
        'mouse' if _is_mouse(organism) else
        'rat' if _is_rat(organism) else
        'human'
    )

    _omnipath_check_version()
    op = _check_if_omnipath()

    # Load Dorothea
    do = op.interactions.Dorothea.get(
        fields=['dorothea_level', 'extra_attrs'],
        dorothea_levels=['A', 'B', 'C', 'D'],
        genesymbols=True,
        organism=_organism,
    )

    # Filter extra columns
    do = do[[
        'source_genesymbol', 'target_genesymbol',
        'is_stimulation', 'is_inhibition',
        'consensus_direction', 'consensus_stimulation',
        'consensus_inhibition', 'dorothea_level',
    ]]

    # Remove duplicates
    do = do[~do.duplicated([
        'source_genesymbol',
        'dorothea_level',
        'target_genesymbol',
    ])]

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
    do['weight'] = [
        i.mor / weights[i.dorothea_level]
        for i in do.itertuples()
    ]

    # Filter and rename
    do = do[[
        'source_genesymbol', 'dorothea_level',
        'target_genesymbol', 'weight',
    ]]
    do.columns = ['source', 'confidence', 'target', 'weight']

    # Filter by levels
    do = (
        do[np.isin(do['confidence'], levels)].
        sort_values('confidence').
        reset_index(drop=True)
    )

    if _organism not in ('mouse', 'rat') and not _is_human(organism):

        do = translate_net(
            net=do,
            target_organism=organism,
            columns=('source', 'target'),
            unique_by=('source', 'target'),
            id_type='genesymbol',
        )

    return do.reset_index(drop=True)


def get_collectri(
        organism: str | int = 'human',
        split_complexes=False,
        ) -> pd.DataFrame:
    """
    CollecTRI gene regulatory network.

    Wrapper to access CollecTRI gene regulatory network. CollecTRI is a
    comprehensive resource containing a curated collection of transcription
    factors (TFs) and their target genes. It is an expansion of DoRothEA.
    Each interaction is weighted by its mode of regulation (either positive or negative).

    Parameters
    ----------
    organism : str
        The organism of interest: either NCBI Taxonomy ID, common name,
        latin name or Ensembl name. Organisms other than human will be
        translated from human data by orthology.
    split_complexes : bool
        Whether to split complexes into subunits. By default complexes are kept as they are.

    Returns
    -------
    ct : DataFrame
        Dataframe in long format containing target genes for each TF with
        their associated weights.
    """

    _organism = (
        'mouse' if _is_mouse(organism) else
        'rat' if _is_rat(organism) else
        'human'
    )

    op = _check_if_omnipath()

    # Load collectri
    ct = op.interactions.CollecTRI.get(genesymbols=True, organism=_organism)

    # Separate gene_pairs from normal interactions
    msk = np.array([s.startswith('COMPLEX') for s in ct['source']])
    cols = ['source_genesymbol', 'target_genesymbol', 'is_stimulation', 'is_inhibition']
    ct_inter = ct.loc[~msk, cols]
    ct_cmplx = ct.loc[msk, cols].copy()

    # Merge gene_pairs into complexes
    if not split_complexes:
        cmpl_gsym = []
        for s in ct_cmplx['source_genesymbol']:
            if s.startswith('JUN') or s.startswith('FOS'):
                cmpl_gsym.append('AP1')
            elif s.startswith('REL') or s.startswith('NFKB'):
                cmpl_gsym.append('NFKB')
            else:
                cmpl_gsym.append(s)
        ct_cmplx.loc[:, 'source_genesymbol'] = cmpl_gsym

    # Merge
    ct = pd.concat([ct_inter, ct_cmplx])

    # Drop duplicates
    ct = ct.drop_duplicates(['source_genesymbol', 'target_genesymbol'])

    # Add weight
    ct['weight'] = [+1 if is_stim else -1 for is_stim in ct['is_stimulation']]

    # Select and rename columns
    ct = ct.rename(columns={'source_genesymbol': 'source', 'target_genesymbol': 'target'})
    ct = ct[['source', 'target', 'weight']]

    if _organism not in ('mouse', 'rat') and not _is_human(organism):

        ct = translate_net(
            net=ct,
            target_organism=organism,
            columns=('source', 'target'),
            unique_by=('source', 'target'),
            id_type='genesymbol',
        )

    return ct.reset_index(drop=True)


def translate_net(
        net: pd.DataFrame,
        columns: str | Iterable[str] = ('source', 'target', 'genesymbol'),
        source_organism: str | int = 'human',
        target_organism: str | int = 'mouse',
        id_type: str | tuple[str, str] = 'genesymbol',
        unique_by: Iterable[str] | None = ('source', 'target'),
        **kwargs: dict[str, str]
        ) -> pd.DataFrame:
    """
    Translate networks between species by orthology.

    This function downloads orthology databases from omnipath and converts
    genes between species. The first time you run this function will take a
    while (~15 minutes) but then it stores all the information in cache for
    quick reusability.

    In case you need to reset the cache, you can do it by doing:
    ``rm -r ~/.pypath/cache/``.

    With its default parameters, this function translates almost any network
    or annotation data frame acquired by the functions in this module from
    human to mouse. For the PROGENy resource you should pass ``columns =
    "target"``, as here the `source` column contains pathways, not identifers.

    Parameters
    ----------
    net : DataFrame
        Network in long format.
    columns : str | list[str] | dict[str, str] | None
        One or more columns to be translated. These columns must contain
        identifiers of the source organism. It can be a single column name,
        a list of column names, or a dict with column names as keys and the
        type of identifiers in these columns as values.
    source_organism: int | str
        Name or NCBI Taxonomy ID of the organism to translate from.
    target_organism : int | str
        Name or NCBI Taxonomy ID of the organism to translate to.
    id_type: str | tuple[str, str]
        Shortcut to provide a single identifier type if all columns
        should be translated from and to the same ID type. If a tuple of two
        provided, the translation happens from the first ID type and the
        orthologs returned in the second ID type.
    kwargs: str | tuple[str, str]
        Alternative way to pass ``columns``. The only limitation is that
        column names can not match any of the existing arguments of this
        function.

    Returns
    -------
    hom_net : DataFrame
        Network in long format with translated genes.
    """

    # Check if pypath is installed
    try:

        def ver(v):
            return tuple(map(int, v.split('.')))

        import pypath
        from pypath.utils import homology
        from pypath.share import common
        from pypath.utils import taxonomy

        if (
            getattr(pypath, '__version__', None) and
            ver(pypath.__version__) < ver(PYPATH_MIN_VERSION)
        ):

            raise RuntimeError(
                'The installed version of pypath-omnipath is too old, '
                f'the oldest compatible version is {PYPATH_MIN_VERSION}.'
            )

    except Exception:

        raise ImportError(
            'pypath-omnipath is not installed. Please install it with: '
            'pip install git+https://github.com/saezlab/pypath.git'
        )

    _source_organism = taxonomy.ensure_ncbi_tax_id(source_organism)
    _target_organism = taxonomy.ensure_ncbi_tax_id(target_organism)

    assert _source_organism, f'Unknown organism: `{source_organism}`.'
    assert _target_organism, f'Unknown organism: `{target_organism}`.'

    if _source_organism == _target_organism:

        return net

    if not isinstance(columns, dict):

        columns = common.to_list(columns)
        columns = dict(zip(columns, [id_type] * len(columns)))

    columns.update(kwargs)
    columns = {k: v for k, v in columns.items() if k in net.columns}

    # Make a copy of net
    hom_net = net.copy()

    # Translate
    hom_net = homology.translate_df(
        df=hom_net,
        target=_target_organism,
        cols=columns,
        source=_source_organism,
    )

    unique_by = common.to_list(unique_by)

    if unique_by and all(c in hom_net.columns for c in unique_by):

        # Remove duplicated based on source and target
        hom_net = hom_net[~hom_net.duplicated(unique_by)]

    return hom_net


def get_ksn_omnipath(
        organism: str | int = 'human',
        ) -> pd.DataFrame:
    """
    OmniPath kinase-substrate network

    Wrapper to access the OmniPath kinase-substrate network. It contains a collection of
    kinases and their target phosphosites. Each interaction is is weighted by its mode of
    regulation (either positive for phosphorylation or negative for dephosphorylation).

    Parameters
    ----------
    organism : str
        The organism of interest: either NCBI Taxonomy ID, common name,
        latin name or Ensembl name. Organisms other than human will be
        translated from human data by orthology. Only human, mouse and rat
        are available for this network.

    Returns
    -------
    ksn : DataFrame
        Dataframe in long format containing target phosphosites for each kinase with
        their associated weights.
    """

    _organism = (
        'mouse' if _is_mouse(organism) else
        'rat' if _is_rat(organism) else
        'human'
    )

    if _organism not in ('mouse', 'rat') and not _is_human(organism):
        raise ValueError('Organism not supported. Only human, mouse and rat are available for this network.')

    op = _check_if_omnipath()

    # Load Kinase-Substrate Network
    ksn = op.requests.Enzsub.get(genesymbols=True, organism=_organism)

    # Filter by phosphorilation
    cols = ['enzyme_genesymbol', 'substrate_genesymbol', 'residue_type', 'residue_offset', 'modification']
    msk = np.isin(ksn['modification'], ['phosphorylation', 'dephosphorylation'])
    ksn = ksn.loc[msk, cols]

    # Build target gene substrate column
    ksn['target'] = ['{0}_{1}{2}'.format(sub, res, off) for sub, res, off in
                     zip(ksn['substrate_genesymbol'], ksn['residue_type'], ksn['residue_offset'])]

    # Assigns mode of regulation
    ksn['weight'] = [+1 if mod == 'phosphorylation' else -1 for mod in ksn['modification']]

    # Remove duplicates
    ksn = ksn.rename(columns={'enzyme_genesymbol': 'source'})[['source', 'target', 'weight']]
    ksn = ksn.drop_duplicates(['source', 'target', 'weight'])

    # If duplicates remain, keep dephosphorylation
    ksn = ksn.groupby(['source', 'target']).min().reset_index()

    return ksn


def _omnipath_check_version() -> None:

    import omnipath

    version = tuple(map(int, omnipath.__version__.split('.')))

    if version < (1, 0, 7):

        day = omnipath.__full_version__.split('.')[-1]

        if day < 'd20230530':

            warnings.warn(
                'The installed version of `omnipath` is older than 1.0.7 or '
                '2023-05-30. To make sure CollecTRI and DoRothEA data is '
                'processed correctly, please update to the latest version by '
                '`pip install git+https://github.com/saezlab/omnipath.git`.'
            )
