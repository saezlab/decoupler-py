"""
Utility functions.
Functions of general utility used in multiple places. 
"""

import numpy as np
import pandas as pd

from .pre import rename_net, get_net_mat

from anndata import AnnData


def m_rename(m, name):
    # Rename
    m = m.rename({'index':'sample', 'variable':'source'}, axis=1)

    # Assign score or pval
    if 'pval' in name:
        m = m.rename({'value':'pval'}, axis=1)
    else:
        m = m.rename({'value':'score'}, axis=1)
    
    return m


def melt(df):
    """
    Function to generate a long format dataframe similar to the one obtained in
    the R implementation of decoupleR.
    
    Parameters
    ----------
    df : dict, tuple, list or pd.DataFrame
        Output of decouple, of an individual method or an individual dataframe.
    
    Returns
    -------
    m : melted long format dataframe.
    """
    
    # If input is result from decoule function
    if type(df) is list or type(df) is tuple:
        df = {k.name:k for k in df}
    if type(df) is dict:
        # Get methods run
        methods = np.unique([k.split('_')[0] for k in df])
        
        res = []
        for methd in methods:
            for k in df:
                # Extract pvals from this method
                pvals = df[methd+'_pvals'].reset_index().melt(id_vars='index')['value'].values
                
                # Melt estimates
                if methd in k and 'pvals' not in k:
                    m = df[k].reset_index().melt(id_vars='index')
                    
                    m = m_rename(m, k)
                    if 'estimate' not in k:
                        name = methd +'_'+k.split('_')[1]
                    else:
                        name = methd
                    m['method'] = name
                    m['pval'] = pvals
                    
                    res.append(m)
        
        # Concat results
        m = pd.concat(res)
            
    # If input is an individual dataframe
    elif type(df) is pd.DataFrame:
        # Melt
        name = df.name
        m = df.reset_index().melt(id_vars='index')
        
        # Rename
        m = m_rename(m, name)
    
    else:
        raise ValueError('Input type {0} not supported.'.format(type(df)))

    return m


def show_methods():
    """
    Shows the available methods.
    The first column correspond to the function name in decoupleR and the 
    second to the method's full name.
    
    Returns
    -------
    df : dataframe with the available methods.
    """
    
    import decoupler
    
    df = []
    lst = dir(decoupler)
    for m in lst:
        if m.startswith('run_'):
            name = getattr(decoupler, m).__doc__.split('\n')[1].lstrip()
            df.append([m, name])
    df = pd.DataFrame(df, columns=['Function', 'Name'])
    
    return df


def check_corr(net, source='source', target='target', weight='weight'):
    """
    Check correlation (colinearity).
    
    Checks the correlation across the regulators in a network.
    
    Parameters
    ----------
    net : pd.DataFrame
        Network in long format.
    source : str
        Column name with source nodes.
    target : str
        Column name with target nodes.
    weight : str
        Column name with weights.
    
    Returns
    -------
    corr : Correlation pairs dataframe.
    """
    
    # Transform net
    net = rename_net(net, source=source, target=target, weight=weight)
    sources, targets, net = get_net_mat(net)
    
    # Compute corr
    corr = np.round(np.corrcoef(net, rowvar=False), 4)
    
    # Filter upper diagonal
    corr = pd.DataFrame(np.triu(corr, k=1), index=sources, columns=sources).reset_index()
    corr = corr.melt(id_vars='index').rename({'index':'source1', 'variable':'source2', 'value':'corr'}, axis=1)
    corr = corr[corr['corr'] != 0]
    
    # Sort by abs value
    corr = corr.iloc[np.argsort(np.abs(corr['corr'].values))[::-1]].reset_index(drop=True)
    
    return corr


def get_acts(adata, obsm_key):
    """
    Extracts activities as AnnData object.
    
    From an AnnData object with source activities stored in `.obsm`,
    generates a new AnnData object with activities in X. This allows
    to reuse many scanpy visualization functions.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with activities stored in .obsm.
    obsm_key
        `.osbm` key to extract.
    
    Returns
    -------
    New AnnData object with activities in X.
    """
    
    obs = adata.obs
    var = pd.DataFrame(index=adata.obsm[obsm_key].columns)
    uns = adata.uns
    obsm = adata.obsm

    return AnnData(np.array(adata.obsm[obsm_key]), 
                       obs=obs, 
                       var=var, 
                       uns=uns,
                       obsm=obsm,
                      )


def get_toy_data(n_samples=24, seed=42):
    """
    Generate a toy `mat` and `net` for testing.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    seed : int
        Random seed to use.
    
    Returns
    -------
    `mat` and `net` examples.
    """
    
    from numpy.random import default_rng

    # Network model
    net = pd.DataFrame(
        [

        ['T1', 'G01',  1], 
        ['T1', 'G02',  1], 
        ['T1', 'G03',0.7],

        ['T2', 'G06',  1], 
        ['T2', 'G07',0.5], 
        ['T2', 'G08',  1],

        ['T3', 'G06',-0.5],
        ['T3', 'G07', -3], 
        ['T3', 'G08', -1],
        ['T3', 'G11', 1],

        ],
        columns = ['source', 'target', 'weight']
    )

    # Simulate two population of samples with different molecular values
    rng = default_rng(seed=seed)
    n_features = 12
    n = int(n_samples/2)
    res = n_samples % 2
    row_a = np.array([8,8,8,8,0,0,0,0,0,0,0,0])
    row_b = np.array([0,0,0,0,8,8,8,8,0,0,0,0])
    row_a = [row_a + np.abs(rng.normal(size=n_features)) for _ in range(n)]
    row_b = [row_b + np.abs(rng.normal(size=n_features)) for _ in range(n+res)]
    
    mat = np.vstack([row_a, row_b])
    features = ['G{:02d}'.format(i+1) for i in range(n_features)]
    samples = ['S{:02d}'.format(i+1) for i in range(n_samples)]
    mat = pd.DataFrame(mat, index=samples, columns=features)
    
    return mat, net
