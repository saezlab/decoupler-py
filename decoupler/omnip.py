import pandas as pd
import numpy as np


def get_progeny(top=100):
    """
    Pathway RespOnsive GENes for activity inference (PROGENy).
    
    Wrapper to access PROGENy model gene weights. Each pathway is defined
    with a collection of target genes, each interaction has an associated
    p-value and weight. The top significant interactions per pathway are 
    returned.
    
    Parameters
    ----------
    top : int
        Number of genes per pathway to return.
    
    Returns
    -------
    DataFrame in long format containing target genes for each pathway with
    their associated weights and p-values.
    """
    
    import omnipath as op
    
    p = op.requests.Annotations.get(resources='PROGENy')
    p = p.set_index(['record_id', 'uniprot', 'genesymbol', 'entity_type', 'source', 'label'])
    p = p.unstack('label').droplevel(axis = 1, level = 0).reset_index()
    p.columns.name = None
    p = p[['genesymbol','p_value','pathway','weight']]
    p = p.sort_values('p_value').groupby('pathway').head(top).reset_index()
    p = p[['pathway','genesymbol','weight','p_value']]
    p['weight'] = p['weight'].astype(np.float32)
    p['p_value'] = p['p_value'].astype(np.float32)
    p.columns = ['source','target','weight','p_value']
    
    return p


def get_resource(name):
    """
    Wrapper to access resources inside Omnipath.
    
    This wrapper allows to easly query different prior knowledge resources. To check
    available resources run [decoupler.show_resources]. For more information visit
    the official website for [Omnipath](https://omnipathdb.org/).
    
    Parameters
    ----------
    name : str
        Name of the resource to query.
    
    Returns
    -------
    DataFrame in long format relating genes to biological entities.
    """
    
    import omnipath as op
    
    resources = show_resources()
    msg = '{0} is not a valid resource. Please, run decoupler.show_resources to see the list of available resources.'
    assert name in resources, msg.format(name)
    
    df = op.requests.Annotations.get(resources=name)
    df = df.set_index(['record_id', 'uniprot', 'genesymbol', 'entity_type', 'source', 'label'])
    df = df.unstack('label').droplevel(axis = 1, level = 0)
    df = df.drop(columns=[name for name in df.index.names if name in df.columns]).reset_index()
    df = df.drop(columns=['record_id','uniprot','entity_type','source'])
    return df


def show_resources():
    """
    Shows the available resources in Omnipath. For more information visit
    the official website for [Omnipath](https://omnipathdb.org/).
    
    Returns
    -------
    List of available resources to query with [decoupler.get_resource].
    """
    import omnipath as op
    return list(op.requests.Annotations.resources())
