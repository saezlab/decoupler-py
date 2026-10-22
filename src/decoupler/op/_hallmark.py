from concurrent.futures import Future, ThreadPoolExecutor

import pandas as pd

from decoupler._docs import docs
from decoupler._download import _bytes_to_pandas, _download
from decoupler.op._dtype import _infer_dtypes
from decoupler.op._translate import translate


@docs.dedent
def hallmark(
    organism: str = "human",
    license: str = "academic",
    verbose: bool = False,
    as_future: bool = False,
) -> pd.DataFrame | Future:
    """
    Hallmark gene sets :cite:p:`msigdb`.

    Hallmark gene sets summarize and represent specific well-defined
    biological states or processes and display coherent expression.

    Parameters
    ----------
    %(organism)s
    %(license)s
    %(verbose)s
    future : bool
        If True, returns a `Future` to allow asynchronous execution.

    Returns
    -------
    Dataframe in long format containing the hallmark gene sets
    or a Future that resolves to it.

    Example
    -------
    .. code-block:: python

        import decoupler as dc

        hm = dc.op.hallmark()
        hm

        # Asynchronous
        future = dc.op.hallmark(as_future=True)
        hm = future.result()
    """

    def _task():
        url = "https://static.omnipathdb.org/tables/msigdb-hallmark.tsv.gz"
        hm = _download(url, verbose=verbose)
        hm = _bytes_to_pandas(hm, sep="\t", compression="gzip")
        hm = hm[["geneset", "genesymbol"]]
        hm["geneset"] = hm["geneset"].str.replace("HALLMARK_", "")
        hm["genesymbol"] = hm["genesymbol"].str.replace("COMPLEX:", "").str.split("_")
        hm = hm.explode("genesymbol")
        hm = _infer_dtypes(hm)
        if organism != "human":
            hm = translate(hm, columns=["genesymbol"], target_organism=organism, verbose=verbose)
        hm = hm.rename(columns={"geneset": "source", "genesymbol": "target"})
        hm = hm.drop_duplicates(["source", "target"]).reset_index(drop=True)
        return hm

    if as_future:
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(_task)
    return _task()
