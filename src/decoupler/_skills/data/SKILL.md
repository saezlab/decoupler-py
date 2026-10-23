---
name: decoupler
description: Read before any task that mentions decoupler, one of its bundled example datasets, or asks which transcription factors (TFs), pathways, or gene sets are most active, enriched, up or down in expression data - bulk RNA-seq treated vs control, single-cell cell types or clusters, condition contrasts across donors (pseudobulk), spatial transcriptomics regions or niches, and pseudotime or differentiation trajectories - with the exact 2.x workflow (dc.op networks, dc.pp preprocessing, dc.mt scoring, dc.tl ranking) that reproduces the documented results. Also covers decoupler's networks (CollecTRI, PROGENy, DoRothEA, MSigDB), supported organisms/species and ortholog translation, resource licenses, and the conventions for plotting or visualizing its results. Also read for conceptual, best-practice or multiple-choice questions about how an agent should work with decoupler.
---

# decoupler 2.x

decoupler scores "sources" (TFs, pathways, gene sets) from a long-format `net` DataFrame with
columns `source`, `target`, and optionally `weight`. The 2.x API is `dc.pp` (preprocess),
`dc.op` (download networks from OmniPath), `dc.mt` (scoring methods), `dc.tl` (ranking),
`dc.pl` (plots), `dc.ds` (example datasets). The 1.x functions (`run_ulm`, `get_progeny`,
`get_acts`) no longer exist.

## Core mechanics (not guessable)

- **Scoring on AnnData writes in place, returns None.** `dc.mt.ulm(data=adata, net=net)` stores
  `adata.obsm["score_ulm"]` and `adata.obsm["padj_ulm"]`. Every method uses `score_<method>` and
  `padj_<method>`, so a second `ulm` call with another network **overwrites** the first.
  Extract before re-running: `score = dc.pp.get_obsm(adata, key="score_ulm")` returns a new
  AnnData whose `.X` is the score matrix (obs = cells, var = sources) and keeps `.obs` and `.obsm`.
- **Scoring on a DataFrame (obs × genes) returns a tuple** `(scores, padj)` of DataFrames.
- **Default method is `ulm`.** Use it unless the user names another. `mt.decouple` runs many
  methods and `mt.consensus` averages them; do not use them as a substitute for `ulm`.
- **Input scale matters.** Methods assume roughly normal values: use log-normalized expression
  (`sc.pp.normalize_total` + `sc.pp.log1p`) or a contrast-level test statistic (the PyDESeq2 Wald
  `stat` column or a t-value). Never score `log2FoldChange` (LFC): it lacks the confidence
  weighting and changes the ranking. Never run on raw integer counts. Do not normalize twice: an `adata.X.max()` far above
  50 means raw counts; a max around 5–12 with non-integer values means already log-normalized.
- **Bundled datasets (`dc.ds.*`) come ready to use.** When a task names a decoupler example
  dataset, load it with `dc.ds.<name>()`; do not substitute another package's copy. The
  single-cell and spatial ones are **already log-normalized and annotated** (obs `celltype`
  and similar): do not run `sc.pp.normalize_total` or `sc.pp.log1p` again, and use the existing
  annotation column instead of re-clustering or hand-annotating. The bulk and pseudobulk ones
  hold raw counts for the contrast recipe. List in `references/resources.md`.
- **Load networks with defaults.** `dc.op.collectri(organism=...)` for TFs,
  `dc.op.progeny(organism=...)` for pathways, `dc.op.hallmark(organism=...)` for MSigDB hallmarks.
  In 2.x `progeny` already defaults to all significant targets (`top=inf`); do **not** pass
  `top=500` (a 1.x habit) unless the user asks. CollecTRI is the documented TF network; use
  DoRothEA only when the user names it.
- **Ranking helpers return sorted DataFrames.**
  `dc.tl.rankby_group(adata=score, groupby=col, reference="rest", method="t-test_overestim_var")`
  gives columns `group, reference, name, stat, meanchange, pval, padj`, sorted by `stat`
  descending within each group. "Most active in group G" = `df[df.group == G]` top rows by `stat`
  (positive = higher in G). `dc.tl.rankby_order(adata=score, order=col)` ranks sources by
  association with a continuous ordering (default `stat="dcor"`, distance correlation, sorted
  descending). Keep the defaults unless asked otherwise.
- **Pass the score AnnData (not the expression AnnData) to the ranking helpers.**

## Pick the recipe from the goal

| Goal | Recipe |
|---|---|
| Activity per cell, cluster or cell type, then "which sources mark group X" | **Per-observation** below |
| Activity per spatial spot, region vs rest | **Spatial** below (smooth first) |
| Sources that change along a trajectory or pseudotime | **Trajectory** below |
| Condition A vs B in bulk RNA-seq (samples) | **Contrast** in `references/contrast.md` |
| Condition A vs B in single-cell with several donors or samples | **Pseudobulk contrast** in `references/contrast.md` |
| Networks, organisms, orthologs, licenses, what resources exist | `references/resources.md` |
| Plotting results | Use `dc.pl.*` or scanpy plotting on the `get_obsm` AnnData, see `references/resources.md` |

"Most active TFs in condition X vs Y" with replicate samples or donors means the contrast recipe
(differential expression statistic → `ulm`), **not** per-cell scores averaged by condition and
**not** `rankby_group` on the condition column. Per-observation scoring + `rankby_group` is for
comparing groups of cells within one dataset (cell types, clusters, niches). The contrast recipe,
complete (details and plots in `references/contrast.md`):

```python
from pydeseq2.dds import DeseqDataSet, DefaultInference
from pydeseq2.ds import DeseqStats
# single-cell with donors only: aggregate raw counts to pseudobulk, QC, pick the cell type
pdata = dc.pp.pseudobulk(adata=adata, sample_col="<donor col>", groups_col="<celltype col>", mode="sum")
dc.pp.filter_samples(pdata, min_cells=10, min_counts=1000)
sub = pdata[pdata.obs["<celltype col>"] == "<cell type>"].copy()
# bulk and pseudobulk: gene filtering (defaults shown) -> PyDESeq2 Wald test -> ulm on `stat`
dc.pp.filter_by_expr(adata=sub, group="<condition col>", min_count=10, min_total_count=15)
dc.pp.filter_by_prop(adata=sub, min_prop=0.1, min_smpls=2)        # pseudobulk only
inference = DefaultInference(n_cpus=8)
# bulk two-group design: "~<condition>". Pseudobulk: print(sub.obs.columns) and add every
# sample-level covariate it holds (e.g. "~<condition> + sex"); writing "~<condition>" alone
# when such a covariate exists gives a different ranking.
dds = DeseqDataSet(adata=sub, design="~<condition> + <covariate>", refit_cooks=True, inference=inference)
dds.deseq2()
stat_res = DeseqStats(dds, contrast=["<condition col>", "<test level>", "<reference level>"], inference=inference)
stat_res.summary()                                # required: fills stat_res.results_df
results_df = stat_res.results_df                  # columns incl. log2FoldChange, stat, padj
# score the Wald `stat` column only, never log2FoldChange and never stat_res.LFC
acts, padj = dc.mt.ulm(data=results_df[["stat"]].T, net=dc.op.collectri(organism="human"))
ranked = acts.T.iloc[:, 0].sort_values(ascending=False)   # most active in <test level> first
```

### Per-observation (single-cell, clusters, cell types)

```python
import scanpy as sc, decoupler as dc
adata = ...                      # log-normalized expression, genes as symbols
net = dc.op.progeny(organism="human")        # or dc.op.collectri(...)
dc.mt.ulm(data=adata, net=net)
score = dc.pp.get_obsm(adata, key="score_ulm")
df = dc.tl.rankby_group(adata=score, groupby="celltype", reference="rest",
                        method="t-test_overestim_var")
top = df[df["group"] == "<group>"].sort_values("stat", ascending=False).head(n)
```

### Spatial (Visium-like)

The documented workflow smooths expression over spatial neighbours **before** scoring; skipping
it changes the ranking. Requires coordinates in `adata.obsm["spatial"]` and log-normalized `X`.

```python
dc.pp.knn(adata, key="spatial", bw=100, cutoff=0.1)        # these are the defaults
adata.X = adata.obsp["spatial_connectivities"].dot(adata.X)  # weighted neighbour average
dc.mt.ulm(data=adata, net=dc.op.collectri(organism="human"))
score = dc.pp.get_obsm(adata, key="score_ulm")
df = dc.tl.rankby_group(adata=score, groupby="<region_col>", reference="rest",
                        method="t-test_overestim_var")
```

### Trajectory or pseudotime

"Associated with the trajectory or differentiation" means: compute scanpy DPT pseudotime, score
sources per cell, then `rankby_order` on the pseudotime with its default statistic. Root the
diffusion pseudotime in the **biologically earliest** population: the stem or progenitor label,
and when several progenitor labels are numbered, the lowest-numbered one (print the label list
and choose deliberately; do not take the first label that happens to match a substring). Use the
first cell of that label as `iroot`.

```python
import numpy as np
sc.pp.neighbors(adata, use_rep="X_pca")
sc.tl.diffmap(adata)
sc.pp.neighbors(adata, use_rep="X_diffmap")
adata.uns["iroot"] = np.flatnonzero(adata.obs["celltype"] == "<earliest celltype>")[0]
sc.tl.dpt(adata)                                   # -> adata.obs["dpt_pseudotime"]
dc.mt.ulm(data=adata, net=dc.op.collectri(organism="mouse"))
score = dc.pp.get_obsm(adata, key="score_ulm")
df = dc.tl.rankby_order(adata=score, order="dpt_pseudotime")   # default stat="dcor"
```

## Checks before reporting

- Organism of the network matches the gene symbols (mouse symbols are capitalized, e.g. `Foxa2`).
- Filter by name family with a regex on `df["name"]` (e.g. `str.match(r"^FOX")`) after ranking,
  not by subsetting the network.
- Report `name` values in the ranked order; the `stat` column is the activity ranking.
