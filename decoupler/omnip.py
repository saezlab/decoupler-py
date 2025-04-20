"""
Utility functions to query OmniPath.
Functions to retrieve resources from the meta-database OmniPath.
"""


import os
import numpy as np
import pandas as pd


url_dbs = 'https://omnipathdb.org/annotations?databases='
url_inter = 'https://omnipathdb.org/interactions/?genesymbols=1&'


def _check_if_liana():
    try:
        import liana as li
    except Exception:
        raise ImportError(
            'liana is not installed. Please install it by: '
            '`pip install liana`.'
        )
    return li


def show_resources():
    """
    Shows available resources in Omnipath. For more information visit the
    official website for [Omnipath](https://omnipathdb.org/).

    Returns
    -------
    lst : list
        List of available resources to query with `dc.get_resource`.
    """

    ann = pd.read_csv('https://omnipathdb.org/queries/annotations', sep='\t')
    ann = ann.set_index('argument').loc['databases'].str.split(';')['values']

    return ann


def get_progeny(
    organism='human',
    top=np.inf,
    thr_padj=0.05,
    license='academic',
    **kwargs
    ):
    """
    Pathway RespOnsive GENes for activity inference (PROGENy).

    Wrapper to access PROGENy model gene weights. Each pathway is defined with
    a collection of target genes, each interaction has an associated p-value
    and weight. The top significant interactions per pathway are returned.

    Parameters
    ----------
    organism : str
        The organism of interest. By default human.
    top : int
        Number of genes per pathway to return. By default all of them.
    thr_padj: float
        Significance threshold to trim interactions.
    license: str
        Which license to use, available options are: academic, commercial, or nonprofit.
        By default, is set to academic to retrieve all possible interactions.
    kwargs
        Passed to `decoupler.translate_net`.

    Returns
    -------
    p : DataFrame
        Dataframe in long format containing target genes for each pathway with
        their associated weights and p-values.
    """
    p = pd.read_csv(url_dbs + f'PROGENy&license={license}', sep="\t")
    p = p[['genesymbol', 'label', 'value', 'record_id']]
    p = p.pivot(index=["genesymbol", "record_id"], columns="label", values="value").reset_index()
    p = p[['pathway', 'genesymbol', 'weight', 'p_value']]
    p = p[~p.duplicated(['pathway', 'genesymbol'])]
    p['p_value'] = p['p_value'].astype(np.float32)
    p['weight'] = p['weight'].astype(np.float32)
    p.columns = ['source', 'target', 'weight', 'p_value']
    p = p.infer_objects()
    p = p.convert_dtypes()
    if organism != 'human':
        p = translate_net(
            p,
            columns='target',
            target_organism=organism,
            **kwargs
        )
    p = (
        p.
        sort_values('p_value').
        groupby('source').
        head(top).
        sort_values(['source', 'p_value']).
        reset_index(drop=True)
    )
    p = p.rename(columns={'p_value': 'pval'})
    p = p[p['pval'] < thr_padj]
    
    return p


def get_resource(
        name,
        organism='human',
        license='academic',
        **kwargs
    ):
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
        The organism of interest. By default human.
    license: str
        Which license to use, available options are: academic, commercial, or nonprofit.
        By default, is set to academic to retrieve all possible interactions.
    kwargs
        Passed to `decoupler.translate_net`.

    Returns
    -------
    df : DataFrame
        Dataframe in long format relating genes to biological entities.
    """

    names = show_resources()
    if name not in names:
        raise ValueError(f'name must be one of these: {names}')
    df = pd.read_csv(url_dbs + f'{name}&license={license}', sep="\t")
    labels = df['label'].unique()
    for label in labels:
        if label in df.columns:
            df.loc[df['label'] == label, 'label'] = f'_{label}'
    df = df[['genesymbol', 'label', 'value', 'record_id']]
    df = df.pivot(index=["genesymbol", "record_id"], columns="label", values="value").reset_index()
    df.index.name = ''
    df.columns.name = ''
    cols_to_remove = ['record_id', 'entity_type']
    df = df.drop(columns=[c for c in cols_to_remove if c in df.columns])
    df = df.infer_objects()
    df = df.convert_dtypes()
    if organism != 'human':
        df = translate_net(
            df,
            columns='genesymbol',
            target_organism=organism,
            **kwargs
        )
    return df


def get_dorothea(
        organism='human',
        levels=['A', 'B', 'C'],
        weight_dict= None,
        license='academic',
        **kwargs
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
    organism : int | str
        The organism of interest. By default human.
    levels : list
        List of confidence levels to return. Goes from A to D, A being the
        most confident and D being the less.
    weight_dict : dict
        Dictionary of values to divide the mode of regulation (-1 or 1),
        one for each confidence level. Bigger values will generate weights
        close to zero.
    license: str
        Which license to use, available options are: academic, commercial, or nonprofit.
        By default, is set to academic to retrieve all possible interactions.
    kwargs
        Passed to `decoupler.translate_net`.

    Returns
    -------
    do : DataFrame
        Dataframe in long format containing target genes for each TF with
        their associated weights and confidence level.
    """

    levels = list(levels)
    weights = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
    weights.update(weight_dict or {})

    # Read
    do = pd.read_csv(url_inter + 
                     f'datasets=dorothea&dorothea_levels=A,B,C,D&fields=dorothea_level&license={license}', sep="\t")
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
    # Format
    do = (
        do
        .rename(columns={'source_genesymbol': 'source', 'target_genesymbol': 'target', 'dorothea_level': 'confidence'})
        [['source', 'target', 'weight', 'confidence']]
        .sort_values('confidence')
    )
    do = do[do['confidence'].isin(levels)].reset_index(drop=True)
    if organism != 'human':
        do = translate_net(
            do,
            columns=['source', 'target'],
            target_organism=organism,
            **kwargs
        )
    do = do.infer_objects()
    do = do.convert_dtypes()
    return do


def get_collectri(
        organism='human',
        split_complexes=False,
        license='academic',
        **kwargs
    ):
    """
    CollecTRI gene regulatory network.

    Wrapper to access CollecTRI gene regulatory network. CollecTRI is a
    comprehensive resource containing a curated collection of transcription
    factors (TFs) and their target genes. It is an expansion of DoRothEA.
    Each interaction is weighted by its mode of regulation (either positive or negative).

    Parameters
    ----------
    organism : int | str
        The organism of interest. By default human.
    split_complexes : bool
        Whether to split complexes into subunits. By default complexes are kept as they are.
    license: str
        Which license to use, available options are: academic, commercial, or nonprofit.
        By default, is set to academic to retrieve all possible interactions.
    kwargs
        Passed to `decoupler.translate_net`.

    Returns
    -------
    ct : DataFrame
        Dataframe in long format containing target genes for each TF with
        their associated weights, and if available, the PMIDs supporting
        each interaction.
    """

    ct = pd.read_csv(url_inter + f'datasets=collectri&fields=references&license={license}', sep="\t")
    ct = ct[['source_genesymbol', 'target_genesymbol', 'is_inhibition', 'references']]
    ct.loc[:, 'pmid'] = (
        ct
        .loc[~ct['references'].isna(), "references"]
        .str.findall(r":(\d+)").apply(lambda x: ";".join(sorted(set(x))))
    )
    ct['weight'] = np.where(ct['is_inhibition'], -1, 1)
    ct = ct.rename(columns={'source_genesymbol': 'source', 'target_genesymbol': 'target'})[['source', 'target', 'weight', 'pmid']]
    if organism != 'human':
        ct = translate_net(
            ct,
            columns=['source', 'target'],
            target_organism=organism,
            **kwargs
        )
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
    ct = ct.infer_objects()
    ct = ct.convert_dtypes()
    ct = ct.groupby(['source', 'target'], as_index=False)['weight'].min()
    return ct


def show_organisms():
    """
    Shows available organisms to translate to with ``translate_net``.

    Returns
    -------
    lst : list
        List of available organisms.
    """
    valid_orgs = [
        'anole_lizard',
        'c.elegans',
        'cat',
        'cattle',
        'chicken',
        'chimpanzee',
        'dog',
        'fruitfly',
        'horse',
        'macaque',
        'mouse',
        'opossum',
        'pig',
        'platypus',
        'rat',
        's.cerevisiae',
        's.pombe',
        'xenopus',
        'zebrafish'
    ]
    return valid_orgs

def translate_net(
    net,
    columns=['source', 'target', 'genesymbol'],
    target_organism='mouse',
    min_evidence=3,
    one_to_many=1,
):
    valid_orgs = show_organisms()
    li = _check_if_liana()
    if target_organism not in valid_orgs:
        raise ValueError(f'target_organism must be one of these: {valid_orgs}')
    if isinstance(columns, str):
        columns = [columns]
    columns = [c for c in columns if c in net.columns]
    if not columns:
        raise ValueError(f'columns must be one of these: {net.columns}')
    # Read orthologs
    target_col = f'{target_organism}_symbol'
    if target_organism == 'anole_lizard':
        target_col = 'anole lizard_symbol'
    elif target_organism == 'fruitfly':
        target_col = 'fruit fly_symbol'
    map_df = li.rs.get_hcop_orthologs(
        url=f'https://ftp.ebi.ac.uk/pub/databases/genenames/hcop/human_{target_organism}_hcop_fifteen_column.txt.gz',
        columns=[f'human_symbol', target_col],
        min_evidence=min_evidence
    )
    map_df = map_df.rename(columns={'human_symbol': 'source', target_col: 'target'})
    df = li.rs.translate_resource(
        net,
        map_df=map_df,
        columns=columns,
        replace=True,
        one_to_many=1,
    ).reset_index(drop=True)
    df = df.drop_duplicates(['source', 'target']).reset_index(drop=True)
    return df


def get_ksn_omnipath(
        license='academic',
    ):
    """
    OmniPath kinase-substrate network

    Wrapper to access the OmniPath kinase-substrate network. It contains a collection of
    kinases and their target phosphosites. Each interaction is is weighted by its mode of
    regulation (either positive for phosphorylation or negative for dephosphorylation).

    Parameters
    ----------
    license: str
        Which license to use, available options are: academic, commercial, or nonprofit.
        By default, is set to academic to retrieve all possible interactions.

    Returns
    -------
    ksn : DataFrame
        Dataframe in long format containing target phosphosites for each kinase with
        their associated weights.
    """

    ks = pd.read_csv(f'https://omnipathdb.org/enz_sub?genesymbols=1&fields=references&license={license}', sep="\t")
    ks = ks.rename(columns={'enzyme_genesymbol': 'source', 'substrate_genesymbol': 'target'})
    ks = ks[ks['modification'].isin(['phosphorylation', 'dephosphorylation'])]
    ks['weight'] = [+1 if mod == 'phosphorylation' else -1 for mod in ks['modification']]
    ks.loc[:, 'pmid'] = (
        ks
        .loc[~ks['references'].isna(), "references"]
        .str.findall(r":(\d+)").apply(lambda x: ";".join(sorted(set(x))))
    )
    ks['target'] = [f'{sub}_{res}{off}' for sub, res, off in zip(ks['target'], ks['residue_type'], ks['residue_offset'])]
    ks = ks[['source', 'target', 'weight', 'pmid']]

    # If duplicates remain, keep dephosphorylation
    ks = ks.groupby(['source', 'target'], as_index=False)['weight'].min()

    return ks

