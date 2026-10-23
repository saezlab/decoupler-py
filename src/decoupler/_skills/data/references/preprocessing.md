# Preprocessing (`dc.pp`)

Helpers to get your data into the shape decoupler expects: pseudobulk aggregation,
expression/sample filtering, layer handling, and spatial neighborhoods. Reminder:
methods need **normalized** input, not raw counts (see
[calling-convention.md](calling-convention.md)).

## Pseudobulk (single cell → sample-level)

```python
pdata = dc.pp.pseudobulk(
    adata,
    sample_col="sample",        # obs column identifying each sample/replicate
    groups_col="cell_type",     # obs column to aggregate within (or None)
    mode="sum",                 # "sum" (default), "mean", or a callable/dict
    layer=None, raw=False,      # where to read counts from
)
```

Returns a new AnnData of pseudobulk profiles (one per sample × group). Aggregate
**raw counts** here, then filter and normalize before scoring.

## Filtering

- `dc.pp.filter_by_expr(adata, group=None, min_count=10, min_total_count=15, ...)` —
  edgeR-style filter of lowly-expressed genes on count data (for bulk/pseudobulk).
  In-place by default (`inplace=True`); pass `inplace=False` to get a mask.
- `dc.pp.filter_by_prop(adata, min_prop=0.2, min_smpls=2)` — keep genes expressed in a
  minimum proportion of cells across a minimum number of samples.
- `dc.pp.filter_samples(adata, min_cells=10, min_counts=1000)` — drop low-quality
  pseudobulk samples.

Typical bulk/pseudobulk order: `pseudobulk` → `filter_by_expr` → normalize (e.g.
CPM + log1p) → `dc.mt.<method>`.

## Layers and reading scores back

- `dc.pp.get_obsm(adata, key="score_ulm")` — pull an `.obsm` score matrix into a new
  AnnData (`.X` = scores, `var` = sources). The bridge to scanpy/plotting.
- `dc.pp.swap_layer(adata, key, X_key="X", inplace=False)` — move a layer into `.X`
  (or vice versa) so you can control which matrix a method reads.
- `dc.pp.extract(data, layer=None, raw=False)` — the low-level extractor methods use
  internally; rarely needed directly, but useful to see exactly what matrix/obs/var a
  method will consume.

## Spatial

- `dc.pp.knn(adata, key="spatial", bw=100, max_nn=100)` — build a spatial
  neighbor graph (used before spatially-smoothed analyses).
- `dc.pp.bin_order(adata, order, nbins=100)` — bin observations along a continuous
  ordering (e.g. pseudotime) for trajectory-style summaries.

## Footguns

- **Pseudobulk on counts, score on normalized.** `pseudobulk` should aggregate raw
  counts; normalize the resulting `pdata` before `dc.mt.*`.
- **`filter_by_expr` expects counts**, not log-normalized values — run it before
  normalization.
- **`inplace=True` is the default** for the filters — they mutate `adata` and return
  `None`. Pass `inplace=False` if you want a boolean mask instead.

## Related references

[getting-started.md](getting-started.md) (where preprocessing fits in the flow),
[calling-convention.md](calling-convention.md) (normalization requirement, `layer`/`raw`),
[io-and-outputs.md](io-and-outputs.md) (`get_obsm`).
