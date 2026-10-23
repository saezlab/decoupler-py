---
name: decoupler
description: Use for any task involving the decoupler library — inferring biological activity/enrichment scores from omics data (bulk, single-cell, spatial). Triggers on estimating transcription factor (TF) activity, pathway activity, or gene-set enrichment from an AnnData/DataFrame; running ulm, mlm, ora, gsea, gsva, aucell, viper, zscore, waggr, mdt, udt; multi-method consensus; fetching prior knowledge (CollecTRI, DoRothEA, PROGENy, MSigDB hallmarks, OmniPath); pseudobulk and expression filtering; using bundled example datasets (pbmc3k, covid5k, erygast1k, hsctgfb, msvisium, knocktf, toy) and whether they are raw counts or normalized; ranking/plotting activity scores; or benchmarking methods against ground truth. This is a router skill — read the relevant file under references/ before writing decoupler code, because outputs land in different places and several defaults are non-obvious.
---

# decoupler

`decoupler` estimates biological activities (transcription factors, pathways, gene
sets) from omics data. You give it a **data matrix** (observations × features, e.g.
cells × genes) and a **prior-knowledge network** (which features belong to which
biological program), and it returns an **enrichment score per observation per
program** — all methods sharing one unified interface. Part of scverse; works
directly on `AnnData`, `pandas.DataFrame`, or raw matrices.

This skill is a **router**. Each topic below has a detailed reference file with exact
signatures and footguns. **Read the relevant reference before writing code** — do not
rely on memory of the API, because results are written to different places depending
on the input type and several methods have non-obvious defaults.

## Two cross-cutting concepts (read these first if unsure)

- **[The unified method call](references/calling-convention.md)** — every method is
  called the same way: `dc.mt.<method>(data, net, tmin=5, ...)`. Covers the accepted
  input types (AnnData / DataFrame / `[matrix, obs, var]` **list**), the shared
  arguments (`tmin`, `raw`, `layer`, `empty`, `bsize`), and the big footguns: input
  must be **normalized (e.g. log1p), not raw counts**, and `net` must be a
  **long-format** `source`/`target`/`weight` table.

- **[Where the results go](references/io-and-outputs.md)** — the return type depends
  on the input type. **AnnData → written in place** into `.obsm["score_<method>"]`
  (and `.obsm["padj_<method>"]` if the method tests), returns `None`. **DataFrame /
  list → returns an `(es, pv)` tuple** (`pv` is `None` for non-testing methods). Read
  scores back out of an AnnData with `dc.pp.get_obsm`. **This is the #1 thing agents
  get wrong.**

## Task → reference file

| If the task is… | Read |
|---|---|
| **starting from scratch** — set up an end-to-end run (load data → get a net → score → rank → plot) | [references/getting-started.md](references/getting-started.md) |
| how to **call a method**, what data/net formats are accepted, `tmin`/`raw`/`layer` semantics | [references/calling-convention.md](references/calling-convention.md) |
| **where scores land** and how to read them back (`.obsm` vs `(es, pv)` tuple, `get_obsm`) | [references/io-and-outputs.md](references/io-and-outputs.md) |
| **choosing and configuring a method** — ulm/mlm/ora/gsea/gsva/aucell/viper/zscore/waggr/mdt/udt + their kwargs | [references/methods.md](references/methods.md) |
| getting **prior knowledge** — TF regulons (CollecTRI, DoRothEA), pathways (PROGENy), gene sets (MSigDB hallmarks), OmniPath, `.gmt`, organism translation | [references/priors.md](references/priors.md) |
| **preprocessing** — pseudobulk from single cell, expression/sample filtering, layers, spatial kNN | [references/preprocessing.md](references/preprocessing.md) |
| running **many methods at once** and combining them into a **consensus** | [references/multi-method.md](references/multi-method.md) |
| **ranking** activity scores by group/ordering and **plotting** them (barplot, dotplot, volcano, network) or overlaying them on a **UMAP / spatial H&E** via scanpy | [references/ranking-and-plotting.md](references/ranking-and-plotting.md) |
| **benchmarking** methods/nets against a perturbation ground truth | [references/benchmarking.md](references/benchmarking.md) |
| using a **bundled example dataset** (`dc.ds.*`) — and whether its `.X` is raw counts or already normalized | [references/datasets.md](references/datasets.md) |

## Quick orientation (submodule map)

decoupler exposes everything through namespaced submodules — canonical import is
`import decoupler as dc`:

- `dc.mt` — **methods**: the enrichment/activity algorithms (`ulm`, `mlm`, `ora`,
  `gsea`, `gsva`, `aucell`, `viper`, `zscore`, `waggr`, `mdt`, `udt`) plus
  `dc.mt.decouple` (run several) and `dc.mt.consensus`. `dc.mt.show()` lists them.
- `dc.op` — **prior knowledge** (OmniPath): `collectri`, `dorothea`, `progeny`,
  `hallmark`, `resource`, `translate`.
- `dc.pp` — **preprocessing**: `pseudobulk`, `filter_by_expr`, `filter_by_prop`,
  `filter_samples`, `get_obsm`, `extract`, `swap_layer`, `knn`, `read_gmt`.
- `dc.ds` — **datasets**: `toy` (used in every example: `adata, net = dc.ds.toy()`),
  `pbmc3k`, `covid5k`, and other real examples. **They differ in normalization state**
  (some raw counts, some already log-normalized) — see
  [datasets.md](references/datasets.md) before scoring one.
- `dc.tl` — **tools**: `rankby_group`, `rankby_obsm`, `rankby_order`.
- `dc.pl` — **plotting**: `barplot`, `dotplot`, `volcano`, `network`, `obsm`, etc.
- `dc.bm` — **benchmarking**: `benchmark`, `metric`, `pl`.

## Conventions used throughout

- **Data** = observations × features (cells/samples × genes), **normalized** (log1p),
  not raw counts. **Net** = long-format DataFrame: `source` (program), `target`
  (feature), optional signed `weight`.
- Output score matrices are observations × sources. AnnData results live in `.obsm`
  under `score_<method>` / `padj_<method>`.
- `tmin` (default 5) silently drops sources with fewer than `tmin` targets present in
  the data — lower it for small/toy nets (examples use `tmin=3`).
- `verbose=True` on any method prints the pruning/run log — use it to see how many
  sources survived `tmin`.
- **Plotting**: use `dc.pl.*` for summary plots and scanpy `sc.pl.umap` / `sc.pl.spatial`
  (on a `dc.pp.get_obsm` score AnnData) for embedding/tissue overlays. Never reimplement
  a plot from matplotlib primitives — see
  [ranking-and-plotting.md](references/ranking-and-plotting.md).
