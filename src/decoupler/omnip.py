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
    estimate : -log10 of the obtained p-values.
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


