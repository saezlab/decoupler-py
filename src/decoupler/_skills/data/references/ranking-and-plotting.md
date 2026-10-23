# Ranking and plotting activity scores

After scoring, you usually want to (a) find which programs are characteristic of which
groups, and (b) visualize them. Ranking lives in `dc.tl`, plotting in `dc.pl`.

Most of these operate on the **score matrix as an AnnData** — get one with
`dc.pp.get_obsm` (see [io-and-outputs.md](io-and-outputs.md)):

```python
dc.mt.ulm(adata, net, tmin=3)
acts = dc.pp.get_obsm(adata, key="score_ulm")   # obs × sources AnnData
```

## Ranking (`dc.tl`)

- `dc.tl.rankby_group(acts, groupby, reference="rest", method="t-test_overestim_var")`
  → DataFrame ranking sources per group (which TFs/pathways mark each cell type).
- `dc.tl.rankby_obsm(adata, key, uns_key="rank_obsm")` → rank the sources in an
  `.obsm` score matrix across obs groups; can store into `.uns`.
- `dc.tl.rankby_order(adata, order, stat="dcor")` → associate sources with a
  continuous ordering (e.g. pseudotime) via a dependence statistic.

```python
df = dc.tl.rankby_group(acts, groupby="cell_type")   # columns include group, name, stat, ...
```

## Plotting (`dc.pl`)

Each returns a matplotlib Figure (or `None`). Common ones:

- `dc.pl.barplot(data, name, top=25)` — top scoring sources for one group/contrast.
  `data` is a **wide `DataFrame`** (rows = groups/contrasts in `.index`, cols = sources)
  and `name` is one **row label**, **not** the `get_obsm` AnnData. You get such a frame
  for free when scoring a contrast/pseudobulk matrix (DataFrame input → wide `scores`),
  or from per-cell scores via a group-mean aggregation (see the example below).
- `dc.pl.dotplot(df, x, y, c, s, top=10)` — dots for a **single contrast**: one dot per
  source, positioned at `x`, labelled by `y`, colored by `c`, sized by `s`. **`x`, `c`,
  and `s` must be numeric columns** — `x` is sorted by `|x|` to pick the `top` rows, so a
  categorical `x` errors. It is *not* a sources × groups grid; filter a `rankby_group`
  result to one group first.
- `dc.pl.volcano(data, x, y, ...)` — volcano of a differential result; pass `net` +
  `name` to highlight a program's target genes.
- `dc.pl.network(net, sources=5, targets=10, score=...)` — draw a source→target graph
  (needs `igraph`); overlay `score`/`data` to color nodes.
- `dc.pl.obsm(adata, key="rank_obsm")` — heatmap of a ranked `.obsm` matrix.
- `dc.pl.source_targets(...)`, `dc.pl.order(...)`, `dc.pl.leading_edge(...)` — target-
  level and ordering/GSEA-style views.

```python
# rank cols: group, reference, name, stat, meanchange, pval, padj
df = dc.tl.rankby_group(acts, groupby="cell_type")

# dotplot: ONE group; x/c/s numeric (x sorted by |x| for `top`), y = source label
import numpy as np
sub = df[df["group"] == "B cells"].copy()
sub["sig"] = -np.log10(sub["padj"].clip(lower=2.22e-16))
dc.pl.dotplot(sub, x="stat", y="name", c="meanchange", s="sig", top=10)

# barplot: wide DataFrame (rows=groups, cols=sources), name = a row label
mean_acts = acts.to_df().groupby(acts.obs["cell_type"], observed=True).mean()
dc.pl.barplot(mean_acts, name="B cells", top=15)
```

Column names for `dotplot`/`volcano` must match the columns of the DataFrame you pass
(inspect `df.columns` from `rankby_group` first), and `dotplot`'s `x`/`c`/`s` columns
must be **numeric** — a categorical `x` (e.g. `x="group"`) raises "category cannot
perform absolute".

## Visualizing scores with scanpy (UMAP / spatial / H&E)

`dc.pl.*` has **no UMAP or spatial plot**. To overlay a program's activity on an
embedding or on tissue, do **not** hand-roll the plot and do **not** copy a single score
into `adata.obs`. Instead, turn the scores into a scanpy-ready AnnData with
`dc.pp.get_obsm` and let scanpy do it. `get_obsm` puts the score matrix in `.X` (so each
**source/TF becomes a `var_name`**) while **preserving `obs`, `uns` (H&E images), and
`obsm` (spatial coords)** — exactly what `sc.pl.umap` / `sc.pl.spatial` need.

```python
import scanpy as sc

dc.mt.ulm(adata, net)                          # adata.obsm["score_ulm"]
acts = dc.pp.get_obsm(adata, "score_ulm")      # scores in .X; keeps obs/uns/obsm

# embedding overlay: "RFXAP" is a var_name -> colored from .X (its activity)
sc.pl.umap(acts, color=["RFXAP"], cmap="RdBu_r")

# spatial / H&E (e.g. dc.ds.msvisium): mix var names and obs columns in one call
sc.pl.spatial(acts, color=["RFXAP", "niches"], cmap="RdBu_r", size=1.5, wspace=0.3)
```

`color=["RFXAP", "niches"]` colors the first panel by RFXAP activity (from `.X`) and the
second by the `niches` obs column — scanpy builds the multi-panel figure itself.

### Spatial/scanpy footguns

- **Do NOT pass `ax=` (from your own `plt.subplots`) into `sc.pl.spatial`.** It manages
  its own figure and colorbar, and passing `ax=` silently breaks `vmin`/`vmax`/`vcenter`
  (the color scale collapses, e.g. to ±0.1, and spots may not render). For multiple
  panels pass a **`color=[...]` list** and use `wspace=`/`ncols=` for layout.
- **Activities are signed and often asymmetric** (e.g. −2 to +9). Use a diverging cmap
  (`RdBu_r`). If you want an explicit scale, set a **symmetric robust limit** rather than
  `vcenter=`:
  ```python
  import numpy as np
  lim = float(np.percentile(np.abs(acts[:, "RFXAP"].X), 99))
  sc.pl.spatial(acts, color="RFXAP", cmap="RdBu_r", vmin=-lim, vmax=lim, size=1.5)
  ```
- `dc.ds.msvisium` is **raw counts** — normalize before scoring (see
  [datasets.md](datasets.md)).

## Footguns

- **Use existing plotting functions — never reimplement a plot.** For summary views use
  `dc.pl.*`; for embeddings/tissue use scanpy `sc.pl.umap` / `sc.pl.spatial` (or
  `squidpy.pl.spatial_scatter`). Do not hand-blit the H&E image + a scatter of spots; if
  a scanpy call misbehaves, fix the call (usually a stray `ax=`), don't replace it.
- `dc.pl.*` functions take **DataFrames**, not the raw `get_obsm` AnnData. But the two
  take *different* frames: `barplot` wants a **wide** group×source matrix (aggregate
  per-cell scores with `acts.to_df().groupby(...).mean()`, or use a wide `scores` from a
  contrast run), while `dotplot` wants a **long** rank table (a `rankby_group` result,
  filtered to one group).
- `dc.pl.network` requires `igraph`; install the `full` extra if missing.
- For `dotplot`, the `x`/`y`/`c`/`s` arguments are **column names** — passing values or
  wrong names errors; check `df.columns` from the ranking step. `x`/`c`/`s` must be
  **numeric** (`x` is sorted by `|x|` internally), so a categorical column there fails.

## Related references

[io-and-outputs.md](io-and-outputs.md) (`get_obsm`, score layout),
[getting-started.md](getting-started.md) (full flow),
[methods.md](methods.md) (what the scores mean per method).
