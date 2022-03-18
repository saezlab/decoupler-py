"""
decouple main function.
Code to run methods simultaneously.
"""

import decoupler as dc
from anndata import AnnData

from .consensus import run_consensus


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
            res = run_consensus({k: results[k] for k in results if k in cns_metds})
        else:
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
