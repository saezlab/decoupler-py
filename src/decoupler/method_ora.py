import numpy as np
import pandas as pd

from .pre import extract, match, rename_net

from fisher import pvalue

from tqdm import tqdm


def get_cont_table(obs, exp, n_background=20000):
    """
    Get contingency table for ORA.
    
    Generates a contingency table (TP, FP, FN and TN) for a list of observed
    features against a list of expected features.
    
    Parameters
    ----------
    obs : list, set
        List or set of observed features.
    exp : list, set
        List or set of expected features.
    
    Returns
    -------
    Values for TP, FP, FN and TN.
    """
    
    # Transform to set
    obs, exp = set(obs), set(exp)
    
    # Build table
    TP = len(obs.intersection(exp))
    FP = len(obs.difference(exp))
    FN = len(exp.difference(obs))
    TN = n_background - TP - FP - FN
    
    return TP, FP, FN, TN


def ora(obs, lexp, n_background=20000):
    """
    Over Representation Analysis (ORA).
    
    Computes ORA to infer regulator activities.
    
    Parameters
    ----------
    obs : list, set
        List of observed features.
    lexp : list, pd.Series
        Iterable of collections of expected features.
    
    Returns
    -------
    Array of uncorrected pvalues.
    """
    
    pvals = [pvalue(*get_cont_table(obs, fset, n_background)).right_tail for fset in lexp]
    
    return pvals


def run_ora(mat, net, source='source', target='target', weight='weight', 
            n_up = 50, n_bottom = 0, n_background = 20000):
    """
    Wrapper to run ORA.
    
    Parameters
    ----------
    mat : list, pd.DataFrame or AnnData
        List of [genes, matrix], dataframe (samples x genes) or an AnnData
        instance.
    net : pd.DataFrame
        Network in long format.
    source : str
        Column name with source nodes.
    target : str
        Column name with target nodes.
    weight : str
        Column name with weights.
    n_up : int
        Number of top ranked features to select as observed features.
    n_bottom : int
        Number of bottom ranked features to select as observed features.
    n_background : int
        Integer indicating the background size.
    
    Returns
    -------
    estimate : -log10 of the obtained p-values.
    pvals : p-values of the enrichements.
    """
    
    # Extract sparse matrix and array of genes
    m, c = extract(mat)
    n_up_msk = len(c) - n_up
    
    # Transform net
    net = rename_net(net, source=source, target=target, weight=weight)
    net = net.groupby('source')['target'].apply(set)
    
    # Run ORA
    pvals = []
    for i in tqdm(range(m.shape[0])):
        obs = np.argsort(m[i].A)[0]
        obs = c[(obs >= n_up_msk) | (obs < n_bottom)]
        pvals.append(ora(obs, net, n_background=n_background))
        
    # Transform to df
    pvals = pd.DataFrame(pvals, columns=net.index)
    pvals.name = 'ora_pvals'
    pvals.columns.name = None
    estimate = pd.DataFrame(-np.log10(pvals), columns=net.index)
    estimate.name = 'ora_estimate'
    estimate.columns.name = None
    
    return estimate, pvals
