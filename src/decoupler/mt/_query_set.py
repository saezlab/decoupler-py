import pandas as pd
import scipy.stats as sts
from tqdm.auto import tqdm

from decoupler._docs import docs
from decoupler._log import _log
from decoupler.mt._ora import _oddsr
from decoupler.pp.net import prune


@docs.dedent
def query_set(
    features: list,
    net: pd.DataFrame,
    alternative: str = "greater",
    n_bg: int | float | None = 20_000,
    ha_corr: int | float = 0.5,
    tmin: int | float = 5,
    verbose: bool = False,
):
    """
    Test overlap between a given feature set against a database of sets.

    Parameters
    ----------
    features
        Set of features
    %(net)s
    alternative
        Defines the alternative hypothesis for fisher exact test. Check ``scipy.stats.fisher_exact``.
    %(n_bg)s
    %(ha_corr)s
    %(tmin)s
    %(verbose)s

    Returns
    -------
    Dataframe containing the odds ratio and fisher exact test p-values for the overlap of the given
    features across sets in a network.

    Example
    -------
    .. code-block:: python

        import decoupler as dc

        ct = dc.op.collectri()
        ft = set(ct[ct["source"] == "SMAD4"]["target"])
        dc.mt.query_set(features=ft, net=ct)
    """
    # Validate
    assert hasattr(features, "__iter__") and not isinstance(features, str | bytes), (
        "features must be an iterable collection of items such as a list"
    )
    features_set: set = set(features)
    if n_bg is None:
        n_bg = 0
        m = "query_set - not using n_bg, a feature specific background will be used instead"
        _log(m, level="info", verbose=verbose)
    assert isinstance(n_bg, int | float) and n_bg >= 0, "n_bg must be numeric and positive"
    # Prune
    net = prune(features=None, net=net, tmin=tmin, verbose=verbose)
    # Test each set against given set
    sources = net["source"].unique()
    df = []
    for source in tqdm(sources, disable=not verbose):
        targets = set(net[net["source"] == source]["target"])
        set_a = features_set.intersection(targets)
        set_b = targets.difference(features_set)
        set_c = features_set.difference(targets)
        a = len(set_a)
        b = len(set_b)
        c = len(set_c)
        if n_bg == 0:
            set_u = set_a.union(set_b).union(set_c)
            set_d = set(net["target"]).difference(set_u)
            d = len(set_d)
        else:
            d = int(n_bg - a - b - c)
            assert d >= 0, (
                f"n_bg={n_bg} must be larger or equal than the number of features to test for source={source} "
                f"({a + b + c}), otherwise the contingency table is invalid. Increase n_bg or set n_bg=None "
                "to use a feature specific background"
            )
        od = _oddsr(a=a, b=b, c=c, d=d, ha_corr=ha_corr, log=True)
        _, pv = sts.fisher_exact([[a, b], [c, d]], alternative=alternative)
        df.append([source, od, pv])
    df = pd.DataFrame(df, columns=["source", "stat", "pval"])
    df["padj"] = sts.false_discovery_control(df["pval"], method="bh")
    df = df.sort_values(["padj", "pval"]).reset_index(drop=True)
    return df
