# Prior-knowledge resources, organisms, licenses, plotting

## Networks (`dc.op`)

All return a long DataFrame with `source`, `target`, and `weight` (plus extra metadata columns
that `dc.mt.*` ignores). Verify by running; downloads come from OmniPath and need network access.

| Function | What | Notes |
|---|---|---|
| `dc.op.collectri(organism="human", remove_complexes=False)` | TF → target regulons (CollecTRI), signed weights | Default TF network in all 2.x vignettes |
| `dc.op.dorothea(organism="human", levels=None)` | Older TF regulons with confidence levels A–E | Only when the user asks for DoRothEA |
| `dc.op.progeny(organism="human", top=inf, thr_padj=0.05)` | 14 signalling pathways, weighted footprint genes | `top` defaults to all significant targets; do not add `top=500` |
| `dc.op.hallmark(organism="human")` | MSigDB hallmark gene sets, unweighted | |
| `dc.op.resource(name, organism="human")` | Any OmniPath annotation resource (e.g. `"PanglaoDB"`, `"SIGNOR"`, `"MSigDB"`) | You must rename columns to `source` and `target` yourself |
| `dc.op.show_resources()` | DataFrame `name, license` of every resource | |

Custom gene sets: `dc.pp.read_gmt(path)` → `source, target` DataFrame. Any DataFrame with
those columns works as `net`.

## Organisms and orthology

- Resources are stored for **human**. `organism=` other than `"human"` translates human symbols to
  the target via orthology (HCOP). `dc.op.show_organisms()` lists the only supported targets:
  anole_lizard, c.elegans, cat, cattle, chicken, chimpanzee, dog, fruitfly, horse, macaque, mouse,
  opossum, pig, platypus, rat, s.cerevisiae, s.pombe, xenopus, zebrafish.
- Always run `dc.op.show_organisms()` before answering a species question; do not trust memory.
  An organism absent from that list **cannot** be used through `organism=` or `dc.op.translate`;
  the user must map symbols themselves (or use human symbols if their data is already in human
  nomenclature).
- `dc.op.translate(net, columns=None, target_organism="mouse", min_evidence=3, one_to_many=5)`
  translates an existing human network's `source` and `target` columns to a supported organism. It
  is human → target only. Genes without orthologs are dropped; `one_to_many` caps how many target
  genes one human gene may map to.

## Licenses

Every resource carries a license tier (`dc.op.show_resources()` column `license`):
`academic`, `nonprofit`, or `commercial`. Each download function takes `license=` and defaults to
`"academic"`, which returns everything. Passing `license="commercial"` (for company use) or
`"nonprofit"` filters out entries whose source database does not allow that use; for some
resources nothing is left (e.g. `dc.op.resource("HPMR", license="commercial")` is empty).
So: a commercial user can use decoupler, but only the entries and resources permitted under
`license="commercial"`, and should set that argument rather than assume all databases are usable.

## Methods (`dc.mt`)

`dc.mt.show()` lists them: aucell, gsea, gsva, mdt, mlm, ora, udt, ulm, viper, waggr, zscore.
Common signature: `dc.mt.<m>(data, net, tmin=5, raw=False, empty=True, bsize=250000, verbose=False, **method_kwargs)`.
`tmin` drops sources with fewer than `tmin` targets present in the data (lower it, e.g. `tmin=3`,
for small marker sets). `empty=True` removes all-zero observations and, if any were removed on an
AnnData, **returns a repaired AnnData instead of None**. `dc.mt.decouple(data, net, methods="all")`
runs several methods (each stored under its own key); `dc.mt.consensus(adata)` adds
`score_consensus` and `padj_consensus`. `ora` is for unweighted gene lists; `gsea` for ranked lists.

## Ranking (`dc.tl`)

- `rankby_group(adata, groupby, reference="rest", method="t-test_overestim_var")` — methods:
  `"t-test"`, `"t-test_overestim_var"`, `"wilcoxon"`. `reference` may be a group name or list.
- `rankby_order(adata, order, stat="dcor")` — stats: `"dcor"`, `"pearsonr"`, `"spearmanr"`, `"kendalltau"`.
  `order` is an obs column (e.g. `dpt_pseudotime`). `dc.pp.bin_order(adata, order, names, label, nbins=100)`
  bins scores along the order for `dc.pl.order`.
- `rankby_obsm(adata, key="X_pca")` — associates an obsm matrix (PCs or scores) with every obs
  column; result in `adata.uns["rank_obsm"]`, plotted by `dc.pl.obsm`.

## Plotting convention

Use decoupler's own `dc.pl.*` functions, or scanpy plotting on the `dc.pp.get_obsm` AnnData
(`sc.pl.umap(score, color=[...])`, `sc.pl.violin`, `sc.pl.matrixplot`, `sc.pl.spatial`), rather
than hand-rolled matplotlib, unless the user explicitly asks for a custom plot. If a `dc.pl` call
errors, fix the arguments; do not silently fall back to a custom plot. Key functions:
`pl.barplot`, `pl.dotplot`, `pl.volcano`, `pl.network`, `pl.source_targets`, `pl.leading_edge`,
`pl.order`, `pl.order_targets`, `pl.obsm`, `pl.obsbar`, and the QC previews `pl.filter_samples`,
`pl.filter_by_expr`, `pl.filter_by_prop`. They return a `matplotlib.figure.Figure` when called
with `return_fig=True`.

## Bundled example datasets (`dc.ds`)

`hsctgfb` (bulk, raw counts, obs `condition`), `covid5k` (single-cell raw counts with donor
metadata, for pseudobulk), `pbmc3k` (single-cell, already log-normalized, obs `celltype` and `leiden`),
`erygast1k` (mouse, log-normalized, has `X_pca`, obs `celltype`), `msvisium` (Visium,
log-normalized, `obsm["spatial"]`, obs `niches`), `toy()` → `(adata, net)`. All download on
first use.
