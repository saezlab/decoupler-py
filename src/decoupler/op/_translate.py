from itertools import product

import pandas as pd

from decoupler._docs import docs
from decoupler._download import _bytes_to_pandas, _download
from decoupler._log import _log


def show_organisms() -> list:
    """
    Shows available organisms to translate to with ``decoupler.op.translate_net``.

    Returns
    -------
    List of available organisms.

    Example
    -------
    .. code-block:: python

        import decoupler as dc

        dc.op.show_organisms()
    """
    valid_orgs = [
        "anole_lizard",
        "c.elegans",
        "cat",
        "cattle",
        "chicken",
        "chimpanzee",
        "dog",
        "fruitfly",
        "horse",
        "macaque",
        "mouse",
        "opossum",
        "pig",
        "platypus",
        "rat",
        "s.cerevisiae",
        "s.pombe",
        "xenopus",
        "zebrafish",
    ]
    return valid_orgs


def _replace_subunits(
    lst: list,
    my_dict: dict,
    one_to_many: int,
) -> list:
    result: list = []
    for x in lst:
        if x in my_dict:
            value = my_dict[x]
            if not isinstance(value, list):
                value = [value]
            if len(value) > one_to_many:
                result.append([])
            else:
                result.append(value)
        else:
            result.append([])
    return result


def _generate_orthologs(resource, column, map_dict, one_to_many):
    df = resource[[column]].drop_duplicates().set_index(column)
    df["subunits"] = df.index.str.split("_")
    df["subunits"] = df["subunits"].apply(
        _replace_subunits,
        args=(
            map_dict,
            one_to_many,
        ),
    )
    df = df["subunits"].explode().reset_index()
    grouped = df.groupby(column).filter(lambda x: x["subunits"].notna().all()).groupby(column)
    # Generate all possible subunit combinations within each group
    complexes = []
    for name, group in grouped:
        if group["subunits"].isnull().all():
            continue
        subunit_lists = [list(x) for x in group["subunits"]]
        complex_combinations = list(product(*subunit_lists))
        for complex in complex_combinations:
            complexes.append((name, "_".join(complex)))
    # Create output DataFrame
    col_names = ["orthology_source", "orthology_target"]
    result = pd.DataFrame(complexes, columns=col_names).set_index("orthology_source")
    return result


def _translate(
    resource: pd.DataFrame,
    map_dict: dict,
    column: str,
    one_to_many: int,
):
    map_data = _generate_orthologs(resource, column, map_dict, one_to_many)
    resource = resource.merge(map_data, left_on=column, right_index=True, how="left")
    resource[column] = resource["orthology_target"]
    resource = resource.drop(columns=["orthology_target"])
    resource = resource.dropna(subset=[column])
    return resource


@docs.dedent
def translate(
    net: pd.DataFrame,
    columns: str | list | None = None,
    target_organism: str = "mouse",
    min_evidence: int = 3,
    one_to_many: int = 5,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Translates gene symbols from human to a target organism using the HCOP database.

    Check which organisms are available with ``decoupler.op.show_organisms``.

    HCOP is a composite database combining data from various orthology resources.
    It provides a comprehensive set of orthologs among human, mouse, and rat, among many other species.

    If you use this function, please reference the original HCOP paper:
    - Yates, B., Gray, K.A., Jones, T.E. and Bruford, E.A., 2021. Updates to HCOP: the HGNC comparison of orthology
    predictions tool. Briefings in Bioinformatics, 22(6), p.bbab155.

    For more information, please visit the HCOP website: https://www.genenames.org/tools/hcop/,
    or alternatively check the bulk download FTP links page: https://ftp.ebi.ac.uk/pub/databases/genenames/hcop/

    Parameters
    ----------
    %(net)s
    columns
        Column or columns of ``net`` to translate.
    target_organism
        Organism to translate to.
    min_evidence
        Minimum number of evidences to keep the interaction, where evidence is the number of
        orthology resources supporting the interaction.
    one_to_many
        Maximum number of orthologs allowed per gene.

    Returns
    -------
    Translated net.

    Example
    -------
    .. code-block:: python

        import decoupler as dc

        ct = dc.op.collectri()
        mouse_ct = dc.op.translate(ct, target_organism="mouse")
        mouse_ct
    """
    # Validate
    assert isinstance(net, pd.DataFrame), "net must be a pd.DataFrame"
    if columns is None:
        columns = ["source", "target", "genesymbol"]
    elif isinstance(columns, str):
        columns = [columns]
    assert isinstance(columns, str | list), "columns must be str or list"
    columns = [c for c in columns if c in net.columns]
    assert columns, f"columns must be one of these: {net.columns}"
    assert isinstance(target_organism, str), "target_organism must be str"
    valid_orgs = show_organisms()
    assert target_organism in valid_orgs, f"target_organism must be one of these: {valid_orgs}"
    assert isinstance(min_evidence, int | float) and min_evidence > 0, "min_evidence must be numerical and > 0"
    assert isinstance(one_to_many, int | float) and one_to_many > 0, "one_to_many must be numerical and > 0"
    _log(f"Translating to {target_organism}", level="info", verbose=verbose)
    # Handle missmatch lavels from ebi db
    target_col = f"{target_organism}_symbol"
    if target_organism == "anole_lizard":
        target_col = "anole lizard_symbol"
    elif target_organism == "fruitfly":
        target_col = "fruit fly_symbol"
    # Process orthologs
    url = f"https://ftp.ebi.ac.uk/pub/databases/genenames/hcop/human_{target_organism}_hcop_fifteen_column.txt.gz"
    map_df = _download(url, verbose=verbose)
    map_df = _bytes_to_pandas(map_df, low_memory=False, compression="gzip", sep="\t")
    map_df = pd.read_csv(url, sep="\t", low_memory=False)
    map_df["evidence"] = map_df["support"].apply(lambda x: len(x.split(",")))
    map_df = map_df[map_df["evidence"] >= min_evidence]
    map_df = map_df[["human_symbol", target_col]]
    map_df = map_df.rename(columns={"human_symbol": "source", target_col: "target"})
    map_dict = map_df.groupby("source")["target"].apply(list).to_dict()
    for col in columns:
        col_unique = net[col].drop_duplicates()
        prop = col_unique.isin(map_dict).sum() / col_unique.size
        m = f"Successfully translated {prop:.1%} of the genes from column {col}."
        _log(m, level="info", verbose=verbose)
        net = _translate(resource=net, map_dict=map_dict, column=col, one_to_many=one_to_many)
    net = net.reset_index(drop=True)
    _log("Translation finished", level="info", verbose=verbose)
    return net
