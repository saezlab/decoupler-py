"""
decouple main function.
Code to run methods simultaneously. 
"""

import decoupler as dc
from anndata import AnnData

import pandas as pd

from .consensus import run_consensus


def get_wrappers(methods):
    tmp = []
    for method in methods:
        try:
            tmp.append(getattr(dc, 'run_'+method))
        except:
            raise ValueError('Method {0} not available, please run show_methods() to see the list of available methods.'.format(method))
    return tmp


def decouple(mat, net, source='source', target='target', weight='weight',
             methods = None, args = {}, consensus=True, min_n=5, 
             verbose=True, use_raw=True):
    """
    Decouple function.
    
    Runs simultaneously several methods of biological activity inference.
    
    Parameters
    ----------
    mat : list, pd.DataFrame or AnnData
        List of [features, matrix], dataframe (samples x features) or an AnnData
        instance.
    net : pd.DataFrame
        Network in long format.
    source : str
        Column name in net with source nodes.
    target : str
        Column name in net with target nodes.
    weight : str
        Column name in net with weights.
    methods : list, tuple
        List of methods to run.
    args : dict
        A dict of argument-dicts.
    consensus_score : bool
        Boolean whether to run a consensus score between methods. 
        Obtained scores are -log10(p-values).
    min_n : int
        Minimum of targets per source. If less, sources are removed.
    verbose : bool
        Whether to show progress.
    use_raw : bool
        Use raw attribute of mat if present.
    
    Returns
    -------
    results : dict of scores and p-values.
    Returns dictionary of activity estimates and p-values or stores them in 
    `mat.obsm['method_estimate']` and `mat.obsm['method_pvals']` for each 
    method.
    """
    
    # If no methods are provided use all
    if methods is None:
        methods = [method.split('run_')[1] for 
                   method in dc.show_methods()['Function']
                   if method != 'run_consensus']
    
    # Retrieve wrapper functions
    wrappers = get_wrappers(methods)
    
    tmp = mat
    if isinstance(mat, AnnData):
        tmp = pd.DataFrame(mat.X, index=mat.obs.index, columns=mat.var.index)
    
    # Store results
    results = {}
    
    # Run every method
    for methd,f in zip(methods,wrappers):
        
        # Init empty args
        if methd not in args:
            a = {}
        else:
            a = args[methd]

        # Overwrite min_n, verbose and use_raw
        a['min_n'] = min_n
        a['verbose'] = verbose
        a['use_raw'] = use_raw

        # Run method
        res = f(mat=tmp, net=net, source=source, target=target, weight=weight, **a)
        
        if type(res) is not tuple:
            res = [res]

        # Store obtained dfs
        for r in res:
            results[r.name] = r
            
    # Run consensus score
    if consensus:
        res = run_consensus(results)
        
        # Store obtained dfs
        for r in res:
            results[r.name] = r
                
    
    if isinstance(mat, AnnData):
        # Store obtained dfs
        for r in results:
            mat.obsm[r] = results[r]
    else:
        return results
