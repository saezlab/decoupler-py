"""
decouple main function.
Code to run methods simultaneously.
"""

import decoupler as dc
from anndata import AnnData

from .consensus import cons
from .pre import extract


def get_wrappers(methods):
    tmp = []
    for method in methods:
        try:
            tmp.append(getattr(dc, 'run_'+method))
        except Exception:
            raise ValueError("""Method {0} not available, please run show_methods() to see the list of available
            methods.""".format(method))
    return tmp


def run_methods(mat, net, source, target, weight, methods, args, min_n, verbose, use_raw):

    # Retrieve wrapper functions
    wrappers = get_wrappers(methods)

    # Store results
    results = {}

    # Run every method
    for methd, f in zip(methods, wrappers):

        # Init empty args
        if methd not in args:
            a = {}
        else:
            a = args[methd]

        # Overwrite min_n, verbose and use_raw
        a['min_n'], a['verbose'], a['use_raw'] = min_n, verbose, use_raw

        # Check if weight method or not
        is_weighted = 'weight' in f.__code__.co_varnames

        # Run method
        if is_weighted:
            res = f(mat=mat, net=net, source=source, target=target, weight=weight, **a)
        else:
            res = f(mat=mat, net=net, source=source, target=target, **a)

        # Extract for AnnData
        if res is None:
            for name in mat.obsm.keys():
                if name.startswith(methd+'_'):
                    results[name] = mat.obsm[name]

        # Extract for mat or df
        else:
            # If only one estimate
            if type(res) is not tuple:
                res = [res]

            # Store obtained dfs
            for r in res:
                results[r.name] = r

    return results


def parse_methods(methods, cns_metds):

    # If no methods are provided use top performers
    if methods is None:
        methods = ['mlm', 'ulm', 'wsum']
        if cns_metds is None:
            cns_metds = ['mlm_estimate', 'ulm_estimate', 'wsum_norm']
    elif not isinstance(methods, list):
        if methods.lower() == 'all':
            methods = [method.split('run_')[1] for method in dc.show_methods()['Function'] if method != 'run_consensus']
        else:
            methods = [methods]

    return methods, cns_metds


def decouple(mat, net, source='source', target='target', weight='weight', methods=None, args={}, consensus=True,
             cns_metds=None, min_n=5, verbose=True, use_raw=True):
    """
    Decouple function.

    Runs simultaneously several methods of biological activity inference.

    Parameters
    ----------
    mat : list, DataFrame or AnnData
        List of [features, matrix], dataframe (samples x features) or an AnnData instance.
    net : DataFrame
        Network in long format.
    source : str
        Column name in net with source nodes.
    target : str
        Column name in net with target nodes.
    weight : str
        Column name in net with weights.
    methods : list, str
        List of methods to run. If none are provided use weighted top performers (mlm, ulm and wsum). To run all methods set to
        "all".
    args : dict
        A dict of argument-dicts.
    consensus : bool
        Boolean whether to run a consensus score between methods.
    cns_metds : list
        List of estimate names to use for the calculation of the consensus score. If empty it will use all the estimates
        obtained after running the different methods. If methods is also None, it will use mlm, ulm and norm_wsum instead.
    min_n : int
        Minimum of targets per source. If less, sources are removed.
    verbose : bool
        Whether to show progress.
    use_raw : bool
        If mat is AnnData, use its raw attribute.

    Returns
    -------
    results : dict
        Dictionary of activity estimates and p-values. If `mat` is AnnData, results for each method are stored in
        `.obsm['method_estimate']` and if available in `.obsm['method_pvals']`.
    """

    # Parse methods
    methods, cns_metds = parse_methods(methods, cns_metds)

    # Check unparied args with methods
    if verbose:
        for methd in args.keys():
            if methd not in methods:
                print('Method name {0} in args not found in methods, will be ignored.'.format(methd))

    # Run methods
    results = run_methods(mat, net, source, target, weight, methods, args, min_n, verbose, use_raw)

    # Run consensus score
    if consensus:
        if cns_metds is not None:
            if type(cns_metds) is not list:
                cns_metds = [cns_metds]
            res = cons({k: results[k] for k in results if k in cns_metds})
        else:
            res = cons(results)

        # Store obtained dfs
        for r in res:
            results[r.name] = r

    if isinstance(mat, AnnData):
        # Store obtained dfs
        for r in results:
            mat.obsm[r] = results[r]
    else:
        return results


def run_consensus(mat, net, source='source', target='target', weight='weight', min_n=5, verbose=False, use_raw=True):
    """
    Consensus score from top methods.
    
    This consensus score is calculated from the three top performer methods: `ulm`, `mlm` and `wsum_norm`.
    For each of these methods, the obtained activities are transformed into z-scores, first for positive
    values and then for negative ones. These two sets of z-score transformed activities are computed by
    subsetting the values bigger or lower than 0, then by mirroring the selected values into their
    opposite sign and finally calculating a classic z-score. This transformation ensures that values across
    methods are comparable, and that they remain in their original sign (active or inactive). The final
    consensus score is the mean across different methods. A p-value is then estimated from these using a
    cumulative distribution function.

    Parameters
    ----------
    mat : list, DataFrame or AnnData
        List of [features, matrix], dataframe (samples x features) or an AnnData instance.
    net : DataFrame
        Network in long format.
    source : str
        Column name in net with source nodes.
    target : str
        Column name in net with target nodes.
    weight : str
        Column name in net with weights.
    min_n : int
        Minimum of targets per source. If less, sources are removed.
    verbose : bool
        Whether to show progress.
    use_raw : bool
        Use raw attribute of mat if present.

    Returns
    -------
    estimate : DataFrame
        Consensus scores. Stored in `.obsm['consensus_estimate']` if `mat` is AnnData.
    pvals : DataFrame
        Obtained p-values. Stored in `.obsm['consensus_pvals']` if `mat` is AnnData.
    """
    
    # Extract sparse matrix and array of features
    m, r, c = extract(mat, use_raw=use_raw, verbose=verbose)

    if verbose:
        print('Running consensus.')

    # Run top methods
    res = decouple(mat=[m, r, c], net=net, source=source, target=target, weight=weight, min_n=min_n,
                      verbose=verbose, use_raw=use_raw)
    
    # Exctract
    estimate, pvals = res['consensus_estimate'], res['consensus_pvals']

    # AnnData support
    if isinstance(mat, AnnData):
        # Update obsm AnnData object
        mat.obsm[estimate.name] = estimate
        mat.obsm[pvals.name] = pvals
    else:
        return estimate, pvals
