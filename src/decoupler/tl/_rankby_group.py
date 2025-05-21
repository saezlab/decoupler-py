import pandas as pd
import numpy as np
import scipy.stats as sts
from anndata import AnnData

from decoupler._docs import docs


@docs.dedent
def rankby_group(
    adata: AnnData,
    groupby: str,
    reference: str | list = 'rest',
    method: str = 't-test_overestim_var',
) -> pd.DataFrame:
    """
    Rank features for characterizing groups.

    Parameters
    ----------
    %(adata)s
    groupby
        The key of the observations grouping to consider.
    reference
        Reference group or list of reference groups to use as reference.
    method
        Statistical method to use for computing differences between groups.
        Avaliable methods include: ``{'wilcoxon', 't-test', 't-test_overestim_var'}``.

    Returns
    -------
    DataFrame with different features between groups.
    """
    assert isinstance(adata, AnnData), 'adata must be anndata.AnnData'
    assert isinstance(groupby, str) and groupby in adata.obs.columns, \
    'groupby must be str and in adata.obs.columns'
    assert isinstance(reference, (str, list)), 'reference must be str or list'
    methods = {'wilcoxon', 't-test', 't-test_overestim_var'}
    assert isinstance(method, str) and method in methods, \
    f'method must be one of: {methods}'
    
    # Get tf names
    features = adata.var_names
    # Generate mask for group samples
    groups = adata.obs[groupby].unique()
    results = []
    for group in groups:
        # Extract group mask
        g_msk = (adata.obs[groupby] == group).values
        # Generate mask for reference samples
        if reference == 'rest':
            ref_msk = ~g_msk
            ref = reference
        elif isinstance(reference, str):
            ref_msk = (adata.obs[groupby] == reference).values
            ref = reference
        else:
            cond_lst = np.array([(adata.obs[groupby] == r).values for r in reference])
            ref_msk = np.sum(cond_lst, axis=0).astype(bool)
            ref = ', '.join(reference)
        assert np.sum(ref_msk) > 0, f'No reference samples found for {reference}'
        # Skip if same than ref
        if group == ref:
            continue
        # Test differences
        result = []
        for i in np.arange(len(features)):
            v_group = adata.X[g_msk, i]
            v_rest = adata.X[ref_msk, i]
            assert np.all(np.isfinite(v_group)) and np.all(np.isfinite(v_rest)), \
                "adata contains not finite values, please remove them."
            if method == 'wilcoxon':
                stat, pval = sts.ranksums(v_group, v_rest)
            elif method == 't-test':
                stat, pval = sts.ttest_ind_from_stats(
                    mean1=np.mean(v_group),
                    std1=np.std(v_group, ddof=1),
                    nobs1=v_group.size,
                    mean2=np.mean(v_rest),
                    std2=np.std(v_rest, ddof=1),
                    nobs2=v_rest.size,
                    equal_var=False,  # Welch's
                )
            elif method == 't-test_overestim_var':
                stat, pval = sts.ttest_ind_from_stats(
                    mean1=np.mean(v_group),
                    std1=np.std(v_group, ddof=1),
                    nobs1=v_group.size,
                    mean2=np.mean(v_rest),
                    std2=np.std(v_rest, ddof=1),
                    nobs2=v_group.size,
                    equal_var=False,  # Welch's
                )
            mc = np.mean(v_group) - np.mean(v_rest)
            result.append([group, ref, features[i], stat, mc, pval])
        # Tranform to df
        result = pd.DataFrame(
            result,
            columns=['group', 'reference', 'name', 'stat', 'meanchange', 'pval']
        )
        # Correct pvalues by FDR
        result['pval'] = result['pval'].fillna(1)
        result['padj'] = sts.false_discovery_control(result['pval'], method='bh')
        # Sort and save
        result['abs_stat'] = result['stat'].abs()
        result = result.sort_values(['padj', 'pval', 'stat'], ascending=[True, True, False])
        result = result.drop(columns=['abs_stat'])
        results.append(result)
    # Merge
    results = pd.concat(results)
    # Convert to categorical
    results['group'] = results['group'].astype('category')
    results['reference'] = results['reference'].astype('category')
    results['name'] = results['name'].astype('category')
    return results.reset_index(drop=True)
