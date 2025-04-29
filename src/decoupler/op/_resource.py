import pandas as pd

from decoupler._log import _log
from decoupler._download import URL_DBS, _download
from decoupler.op._translate import translate
from decoupler.op._dtype import _infer_dtypes


def show_resources(
) -> list:
    """
    Shows available resources in Omnipath. For more information visit the
    official website for [Omnipath](https://omnipathdb.org/).

    Returns
    -------
    List of available resources to query with `decoupler.op.resource`.
    """
    ann = pd.read_csv('https://omnipathdb.org/queries/annotations', sep='\t')
    ann = ann.set_index('argument').loc['databases'].str.split(';')['values']
    return ann


def resource(
    name: str,
    organism: str = 'human',
    license: str = 'academic',
    verbose: bool = False,
):
    """
    Wrapper to access resources inside Omnipath.

    This wrapper allows to easly query different prior knowledge resources. To
    check available resources run ``decoupler.op.show_resources()``. For more
    information visit the official website for
    [Omnipath](https://omnipathdb.org/).

    Parameters
    ----------
    name:
        Name of the resource to query.
    organism
        The organism of interest.
    license:
        Which license to use, available options are: academic, commercial, or nonprofit.
        By default, is set to academic to retrieve all possible interactions.
    kwargs
        Passed to ``decoupler.op.translate``.

    Returns
    -------
    Network in long format.
    """
    # Validate
    assert isinstance(name, str), 'name must be str'
    names = show_resources()
    assert name in names, f'name must be one of these: {names}'
    assert isinstance(organism, str), 'organism must be str'
    assert isinstance(license, str) and license in ['academic', 'commercial', 'nonprofit'], \
    'license must be academic, commercial or nonprofit'
    assert isinstance(verbose, bool), 'verbose must be bool'
    m = f'Accessing {name} with {license} license'
    _log(m, level='info', verbose=verbose)
    # Download
    url = URL_DBS + f'{name}&license={license}'
    df = _download(url, verbose=verbose)
    # Process
    labels = df['label'].unique()
    for label in labels:
        if label in df.columns:
            df.loc[df['label'] == label, 'label'] = f'_{label}'
    df = df[['genesymbol', 'label', 'value', 'record_id']]
    df = df.pivot(index=["genesymbol", "record_id"], columns="label", values="value").reset_index()
    df.index.name = ''
    df.columns.name = ''
    cols_to_remove = ['record_id', 'entity_type', '_entity_type']
    df = df.drop(columns=[c for c in cols_to_remove if c in df.columns])
    df = _infer_dtypes(df)
    if organism != 'human':
        df = translate(df, columns='genesymbol', target_organism=organism, verbose=verbose)
    return df
