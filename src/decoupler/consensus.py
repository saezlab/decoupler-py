import numpy as np
import pandas as pd

from scipy.stats import beta

from .utils import melt


def rankMatrix(glist):
    u = np.unique(sum(glist, []))
    N = len(u)

    rmat = pd.DataFrame(np.ones((N, len(glist))), index=u)
    N = np.repeat(N, rmat.shape[1])

    for i in range(len(glist)):
        rmat.loc[glist[i], i] = np.arange(1, len(glist[i])+1)/N[i]
    
    return rmat

def correctBetaPvalues(p, k):
    p = np.min([p * k, 1])
    return p

def betaScores(r):
    n = len(r)
    r = np.sort(r)
    p = beta.cdf(r, np.arange(n)+1, n - (np.arange(n)+1) + 1)
    return p

def rhoScores(r):
    x = betaScores(r)
    rho = correctBetaPvalues(np.min(x), k = len(r))
    return rho

def aggregateRanks(rmat):
    df = []
    for name, r in zip(rmat.index, rmat.values):
        pval = rhoScores(r)
        df.append([name, pval])
    df = pd.DataFrame(df, columns=['source', 'pval'])
    return df

def run_consensus(res):
    """
    Consensus.
    
    Runs a consensus score using RobustRankAggreg after running different 
    methods with decouple.
    
    Parameters
    ----------
    res : list, tuple or pd.DataFrame
        Results from `decouple`.
    
    Returns
    -------
    estimate : activity estimates.
    pvals : p-values of the obtained activities.
    """
    
    # Melt res if is dict
    if type(res) is dict:
        res = melt(res)
    
    # Get unique samples and methods
    samples = res['sample'].unique()
    methods = res['method'].unique()

    # Get pval for each sample
    pvals = []
    for sample in samples:
        # Subset by sample and abs scores
        sdf = res[res['sample'] == sample].assign(score=lambda x: abs(x.score))
        
        # Generate list of lists of sorted sources
        lst = []
        for methd in methods:
            sources = sdf[sdf['method'] == methd].sort_values('score', ascending=False)['source']
            lst.append(list(sources))
        
        # Run AggregateRanks
        rmat = rankMatrix(lst)
        rnks = aggregateRanks(rmat)
        rnks['sample'] = sample
        pvals.append(rnks)

    # Transform to df
    pvals = pd.concat(pvals).pivot(index='sample', columns='source', values='pval')
    pvals.name = 'consensus_pvals'
    pvals.columns.name = None
    pvals.index.name = None
    
    # Get estimate
    estimate = pd.DataFrame(-np.log10(pvals), columns=pvals.columns)
    estimate.name = 'consensus_estimate'
    
    return estimate, pvals
