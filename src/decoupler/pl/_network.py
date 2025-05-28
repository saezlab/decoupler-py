import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.gridspec
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from decoupler._odeps import ig, Graph, _check_import
from decoupler._docs import docs
from decoupler._Plotter import Plotter


def _src_idxs(
    score: pd.DataFrame,
    sources: int | list | str,
    by_abs: bool,
) -> np.ndarray:
    assert isinstance(sources, (int, list, str)), "sources must be int, list or str"
    if isinstance(sources, int):
        if by_abs:
            s_idx = np.argsort(-abs(score.values[0]))[:sources]
        else:
            s_idx = np.argsort(-score.values[0])[:sources]
    elif isinstance(sources, list):
        s_idx = np.isin(score.columns.astype(str), sources)
    else:
        s_idx = np.isin(score.columns.astype(str), [sources])
    return s_idx


def _trg_idxs(
    data: pd.DataFrame,
    net: pd.DataFrame,
    targets: int | list | str,
    by_abs: bool,
) -> np.ndarray:
    assert isinstance(targets, (int, list, str)), "targets must be int, list or str"
    if isinstance(targets, int):
        net["prod"] = [
            data.iloc[0][t] * w if t in data.columns else 0 for t, w in zip(net["target"], net["weight"], strict=False)
        ]
        if by_abs:
            net["prod"] = abs(net["prod"])
        t_idx = (
            net.sort_values(["source", "prod"], ascending=[True, False])
            .groupby(["source"], observed=True)
            .head(targets)
            .index.values
        )
    elif isinstance(targets, list):
        t_idx = np.isin(net["target"].astype(str), targets)
    else:
        t_idx = np.isin(net["target"].astype(str), [targets])
    return t_idx


def _filter(
    data: pd.DataFrame,
    score: pd.DataFrame,
    net: pd.DataFrame,
    sources: int,
    targets: int,
    by_abs: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert isinstance(data, pd.DataFrame), "data must be pd.DataFrame"
    assert isinstance(score, pd.DataFrame), "score must be pd.DataFrame"
    assert np.all(data.index == score.index) and (data.index.size == 1), (
        "data and score need to have the same row index."
    )
    assert isinstance(by_abs, bool), "by_abs must be bool"
    # Select top sources
    s_idx = _src_idxs(score=score, sources=sources, by_abs=by_abs)
    # Filter
    score = score.iloc[:, s_idx]
    net = net.loc[np.isin(net["source"].astype(str), score.columns.astype(str)), :].copy()
    if "weight" not in net.columns:
        net["weight"] = 1.0
    # Select top targets
    t_idx = _trg_idxs(data=data, net=net, targets=targets, by_abs=by_abs)
    # Filter
    net = net.loc[t_idx]
    # Filter unmatched features
    data = data.loc[:, np.isin(data.columns.astype(str), net["target"].astype(str))]
    net = net.loc[np.isin(net["target"].astype(str), data.columns.astype(str)), :]
    return data, score, net


def _norm(
    x: np.ndarray,
    vcenter: bool,
) -> matplotlib.colors.Normalize:
    assert isinstance(vcenter, bool), "vcenter must be bool"
    if vcenter:
        vmax = np.max(np.abs(x))
        norm = matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax)
    else:
        vmax = np.max(x)
        vmin = np.min(x)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    return norm


def _dict_types(
    data: pd.DataFrame,
    score: pd.DataFrame,
) -> tuple[dict, np.ndarray]:
    vs = np.unique(np.hstack([data.columns, score.columns]))
    v_dict = {k: i for i, k in enumerate(vs)}
    types = (~np.isin(vs, score.columns)) * 1
    return v_dict, types


def _net_2_elist(
    net: pd.DataFrame,
    v_dict: dict,
) -> list:
    edges = []
    for i in net.index:
        source, target = net.loc[i, "source"], net.loc[i, "target"]
        edge = [v_dict[source], v_dict[target]]
        edges.append(edge)
    return edges


def _net_2_g(
    data: pd.DataFrame,
    score: pd.DataFrame,
    net: pd.DataFrame,
) -> Graph:
    # Unify network
    v_dict, types = _dict_types(data=data, score=score)
    # Transform net to edges
    edges = _net_2_elist(net=net, v_dict=v_dict)
    # Create graph
    g = ig.Graph(
        edges=edges,
        directed=True,
    )
    # Update attributes
    g.es["weight"] = net["weight"].values
    g.vs["type"] = types
    g.vs["label"] = list(v_dict.keys())
    g.vs["shape"] = np.where(types, "circle", "square")
    return g


def _gcolors(
    g: Graph,
    data: pd.DataFrame,
    score: pd.DataFrame,
    s_norm: matplotlib.colors.Normalize,
    t_norm: matplotlib.colors.Normalize,
    s_cmap: str,
    t_cmap: str,
) -> bool:
    cmaps = matplotlib.colormaps.keys()
    if (s_cmap in cmaps) and (t_cmap in cmaps):
        s_cmap = matplotlib.colormaps.get_cmap(s_cmap)
        t_cmap = matplotlib.colormaps.get_cmap(t_cmap)
        color = []
        for i, k in enumerate(g.vs["label"]):
            if g.vs["type"][i]:
                color.append(t_cmap(t_norm(data[k].values[0])))
            else:
                color.append(s_cmap(s_norm(score[k].values[0])))
        is_cmap = True
    else:
        color = [s_cmap if typ == 0.0 else t_cmap for typ in g.vs["type"]]
        is_cmap = False
    g.vs["color"] = color
    return is_cmap


@docs.dedent
def network(
    net,
    data: pd.DataFrame = None,
    score: pd.DataFrame = None,
    sources: int | list | str = 5,
    targets: int | list | str = 10,
    by_abs=True,
    size_node=5,
    size_label=2.5,
    s_cmap="RdBu_r",
    t_cmap="viridis",
    vcenter=False,
    c_pos_w="darkgreen",
    c_neg_w="darkred",
    s_label="Enrichment\nscore",
    t_label="Gene\nexpression",
    layout="kk",
    **kwargs,
):
    """
    Plot results of enrichment analysis as network.

    Parameters
    ----------
    %(net)s
    data
        Input of enrichment analysis, needs to be a one row dataframe with targets as features. Used to filter net.
    score
        Ouput of enrichment analysis, needs to be a one row dataframe with sources as features. Used to filter net.
    sources
        Number of top sources to plot or list of source names.
    targets
        Number of top targets to plot or list of target names.
    by_abs
        Whether to consider the absolute value when sorting for ``n_sources`` and ``n_targets``.
    size_node
        Size of the nodes in the plot.
    size_label
        Size of the labels in the plot.
    s_cmap
        Color or colormap to use to color sources.
    t_cmap
        Color or colormap to use to color targets.
    vcenter
        Whether to center colors around 0.
    c_pos_w
        Color for edges with positive weights. If no weights are available, they are set to positive by default.
    c_neg_w
        Color for edges with negative weights.
    s_label
        Label to place in the source colorbar.
    t_label
        Label to place in the target colorbar.
    layout
        Layout to use to order the nodes. Check ``igraph`` documentation for more options.
    %(plot)s
    """
    assert isinstance(net, pd.DataFrame), "net must be pd.DataFrame"
    assert (data is None) == (score is None), "data and score must either both be None"
    _check_import(ig)
    if data is None:
        srcs = net["source"].unique().astype("U")
        score = pd.DataFrame(np.ones((1, srcs.size)), index=["0"], columns=srcs)
        trgs = net["target"].unique().astype("U")
        data = pd.DataFrame(np.ones((1, trgs.size)), index=["0"], columns=trgs)
        t_cmap = "white"
    # Filter
    fdata, fscore, fnet = _filter(
        data=data,
        score=score,
        net=net,
        sources=sources,
        targets=targets,
        by_abs=by_abs,
    )
    # Color norms
    s_norm = _norm(x=fscore, vcenter=vcenter)
    t_norm = _norm(x=fdata, vcenter=vcenter)
    # Make graph
    g = _net_2_g(data=fdata, score=fscore, net=fnet)
    g.es["color"] = [c_pos_w if w > 0 else c_neg_w for w in g.es["weight"]]
    is_cmap = _gcolors(
        g=g,
        data=data,
        score=score,
        s_norm=s_norm,
        t_norm=t_norm,
        s_cmap=s_cmap,
        t_cmap=t_cmap,
    )
    # Instance
    kwargs["ax"] = None
    bp = Plotter(**kwargs)
    bp.fig.delaxes(bp.ax)
    # Plot
    gs = matplotlib.gridspec.GridSpec(6, 3)
    ax1 = bp.fig.add_subplot(gs[:-1, :])
    ax2 = bp.fig.add_subplot(gs[-1, 0])
    ax3 = bp.fig.add_subplot(gs[-1, 1])
    ax4 = bp.fig.add_subplot(gs[-1, -1])
    ig.plot(
        g,
        target=ax1,
        layout=layout,
        vertex_size=(size_node * bp.dpi) / (bp.figsize[0] * bp.figsize[0]),
        vertex_size_label=(size_label * bp.dpi) / (bp.figsize[0] * bp.figsize[0]),
        bbox_inches="tight",
    )
    if is_cmap:
        sm = matplotlib.cm.ScalarMappable(norm=s_norm, cmap=s_cmap)
        bp.fig.colorbar(sm, cax=ax2, orientation="horizontal", label=s_label)
        sm = matplotlib.cm.ScalarMappable(norm=t_norm, cmap=t_cmap)
        bp.fig.colorbar(sm, cax=ax4, orientation="horizontal", label=t_label)
    else:
        ax2.axis("off")
        ax4.axis("off")
    # Add legend
    square = Line2D(
        [0], [0], marker="s", color="w", label="Source", markerfacecolor="white", markeredgecolor="black", markersize=10
    )
    circle = Line2D(
        [0], [0], marker="o", color="w", label="Target", markerfacecolor="white", markeredgecolor="black", markersize=10
    )
    line1 = Line2D(
        (0, 0),
        (1, 0),
        color=c_pos_w,
        lw=2,
        marker=">",
    )
    line2 = Line2D(
        (0, 0),
        (1, 0),
        color=c_neg_w,
        lw=2,
        marker=">",
    )
    handles = [square, circle, line1, line2]
    labels = ["Source", "Target", "Positive", "Negative"]
    legend = ax3.legend(
        handles=[square, circle, line1, line2],
        labels=labels,
        frameon=False,
        loc="center",
        bbox_to_anchor=(0.5, 0.5),
        bbox_transform=ax3.transAxes,
    )
    ax3.axis("off")
    return bp._return()
