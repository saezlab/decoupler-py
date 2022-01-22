import numpy as np
import pandas as pd

from numpy.random import default_rng

from scipy.stats import beta


def beta_scores(rmat):
    rmat = np.sort(rmat, axis=0)
    n = rmat.shape[0]
    dist_a = np.repeat([np.repeat([np.arange(n)], rmat.shape[1], axis=0)], rmat.shape[2], axis=0).T + 1
    dist_b = n - dist_a + 1
    p = beta.cdf(np.sort(rmat, axis=0), dist_a, dist_b)
    return p


def corr_beta_pvals(p, k):
    p = np.clip(p * k, a_min=0, a_max=1)
    return p


def aggregate_ranks(acts):
    rmat = (acts.shape[-1] - np.argsort(np.argsort(acts))) / (acts.shape[-1])
    x = beta_scores(rmat)
    rho = corr_beta_pvals(np.min(x, axis=0), k = rmat.shape[0])
    return rho


def run_consensus(res, seed=42):
    """
    Consensus.
    
    Runs a consensus score using RobustRankAggreg after running different 
    methods with decouple.
    
    Parameters
    ----------
    res : dict
        Results from `decouple`.
    seed : int
        Random seed to use.
    
    Returns
    -------
    Returns activity estimates (-log10(p-values)) and p-values.
    """
    
    acts = np.abs([res[k].values for k in res if 'pvals' not in k and 
                   not np.all(np.isnan(res[k].values.astype(np.float32)))])
    
    # Randomize sources order to break ties randomly
    rng = default_rng(seed=seed)
    idx = np.arange(acts.shape[2])
    rng.shuffle(idx)
    acts = acts[:,:,idx]
    
    # Compute p-vals
    pvals = aggregate_ranks(acts)
    
    # Transform to df
    k = list(res.keys())[0]
    pvals = pd.DataFrame(pvals, index=res[k].index, columns=res[k].columns[idx])
    pvals.name = 'consensus_pvals'
    pvals.columns.name = None
    pvals.index.name = None
    
    # Get estimate
    estimate = pd.DataFrame(-np.log10(pvals), columns=pvals.columns)
    estimate.name = 'consensus_estimate'
    
    return estimate, pvals
