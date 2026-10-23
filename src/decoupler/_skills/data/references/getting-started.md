# Getting started (end-to-end)

A complete decoupler run has four steps: **get a normalized data matrix**, **get a
prior net**, **score**, then **inspect/rank/plot**. Below are the two most common
end-to-end shapes.

## Minimal, self-contained (toy data)

```python
import decoupler as dc

# 1. Data (obs × var, normalized) + 2. a prior net (source/target/weight)
adata, net = dc.ds.toy()

# 3. Score — results written into adata.obsm in place
dc.mt.ulm(adata, net, tmin=3)

# 4. Inspect: scores + adjusted p-values, both obs × sources
adata.obsm["score_ulm"]
adata.obsm["padj_ulm"]
```

`dc.ds.toy()` returns a small AnnData and a matching weighted net — use it for any
quick check or when reproducing behavior.

## Single-cell TF activity (realistic shape)

```python
import scanpy as sc
import decoupler as dc

# adata: raw counts in a single-cell AnnData
sc.pp.normalize_total(adata, target_sum=1e4)   # when normalizing, use target_sum=1e4
sc.pp.log1p(adata)                      # decoupler needs normalized input, not counts
# NOTE: only normalize raw counts. Data that is already log-normalized (e.g.
# dc.ds.pbmc3k()) must NOT be normalized again — see references/datasets.md.

net = dc.op.collectri(organism="human")  # signed TF regulons

dc.mt.ulm(adata, net)                    # -> adata.obsm["score_ulm"], ["padj_ulm"]

# Turn scores into an AnnData for downstream scanpy/plotting
acts = dc.pp.get_obsm(adata, key="score_ulm")   # obs × TFs

# Which TFs are most active per cell type?
df = dc.tl.rankby_group(acts, groupby="cell_type")
# barplot/dotplot take DataFrames (not `acts`) — see ranking-and-plotting.md
```

## Pathway activity or gene-set enrichment

Swap the net (and, for unweighted gene sets, the method):

```python
# Pathways (weighted) — ulm is fine
prog = dc.op.progeny(organism="human", top=500)
dc.mt.ulm(adata, prog)                   # adata.obsm["score_ulm"] = pathway activities

# Hallmark gene sets (unweighted) — use a gene-set method
hall = dc.op.hallmark(organism="human")
dc.mt.ora(adata, hall)                   # adata.obsm["score_ora"], ["padj_ora"]
```

## Bulk / pseudobulk

For bulk RNA-seq or pseudobulk from single cell, aggregate and filter first (see
[preprocessing.md](preprocessing.md)), then score exactly the same way:

```python
pdata = dc.pp.pseudobulk(adata, sample_col="sample", groups_col="cell_type")
dc.pp.filter_by_expr(pdata)              # drop lowly-expressed genes
# ... normalize pdata (e.g. CPM + log1p) ...
dc.mt.ulm(pdata, net)
```

## Where to go next

- Method choice and kwargs → [methods.md](methods.md)
- Getting/formatting the net → [priors.md](priors.md)
- Return types and reading scores back → [io-and-outputs.md](io-and-outputs.md)
- Running many methods + consensus → [multi-method.md](multi-method.md)
- Ranking + plotting → [ranking-and-plotting.md](ranking-and-plotting.md)
