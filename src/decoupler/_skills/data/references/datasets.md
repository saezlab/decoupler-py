# Bundled example datasets (`dc.ds`)

decoupler ships loaders for a toy dataset and several real ones. **They are not all in
the same state** — some `.X` are raw counts, some are already log-normalized, one is a
log2FC contrast statistic. Since methods expect **normalized** input (see
[calling-convention.md](calling-convention.md)), you must know which is which or you
will either skip a needed normalization or re-normalize already-normalized data.

## Rule: inspect `.X` before assuming counts

```python
import numpy as np, scipy.sparse as sps
X = adata.X.toarray() if sps.issparse(adata.X) else np.asarray(adata.X)
X.max(), np.allclose(X[:50], np.round(X[:50])), "log1p" in adata.uns
```

- **Integer-valued and large max (hundreds–millions)** → raw counts → normalize first.
- **Float, small max (roughly ≤ tens), often `adata.uns["log1p"]` present** → already
  log-normalized → **score directly, do not normalize again**.
- **Signed values (negatives)** → a contrast statistic (e.g. log2FC), not expression →
  score directly.

## The datasets

| Loader | What it is | `.X` state | Before scoring | Organism |
|---|---|---|---|---|
| `dc.ds.toy()` | synthetic AnnData **+ matching net** | synthetic continuous | score directly (`tmin=3`) | — |
| `dc.ds.toy_bench()` | toy + benchmark ground truth | synthetic continuous | feed to `dc.bm` | — |
| `dc.ds.pbmc3k()` | 10x PBMCs (~2.6k cells) | **log-normalized** (max ~6) | **score directly** | human |
| `dc.ds.erygast1k()` | mouse erythroid gastrulation (~800 cells) | **log-normalized** (`uns['log1p']`) | score directly | **mouse** |
| `dc.ds.covid5k()` | COVID-19 PBMCs (~5k cells) | **raw counts** (max ~1e4) | **normalize + log1p** | human |
| `dc.ds.hsctgfb()` | bulk RNA-seq, 6 HSC samples | **raw counts** | **filter + normalize** (bulk) | human |
| `dc.ds.msvisium()` | Visium MS brain slide (spatial) | **raw counts** | **normalize** | human |
| `dc.ds.knocktf()` | KnockTF TF-perturbation contrasts | **log2FC** (−20…20) | score directly (benchmark) | human |

### Ready to score directly (do NOT normalize)

```python
adata = dc.ds.pbmc3k()            # already log-normalized
net = dc.op.collectri(organism="human")
dc.mt.ulm(adata, net)             # -> adata.obsm["score_ulm"], ["padj_ulm"]
```

- `erygast1k` is the same, but **mouse** — fetch a mouse net (`organism="mouse"`) or
  translate a human one (`dc.op.translate`, see [priors.md](priors.md)). Its `obs` has
  `celltype`, `stage`.
- `pbmc3k.obs` has `celltype` and `leiden`; `obsm` has PCA/UMAP/tSNE already.

### Raw counts → normalize first

```python
import scanpy as sc
adata = dc.ds.covid5k()                    # raw integer counts
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
dc.mt.ulm(adata, dc.op.collectri())
```

- `covid5k.obs`: `individual`, `sex`, `disease`, `celltype`.
- `hsctgfb` is **bulk** (6 samples, `obs`: `condition`, `sample_id`) — filter lowly
  expressed genes with `dc.pp.filter_by_expr` then normalize; a natural fit for a
  bulk/DE + activity workflow ([preprocessing.md](preprocessing.md)).
- `msvisium` is **spatial** — normalize, then optionally build a neighbor graph with
  `dc.pp.knn(adata, key="spatial")`. It carries `obsm["spatial"]`, `uns["spatial"]`
  (H&E images), and `obs["niches"]`. To plot activity on the H&E, use `dc.pp.get_obsm` +
  `sc.pl.spatial` (see [ranking-and-plotting.md](ranking-and-plotting.md)).

### Benchmark inputs (not for routine scoring)

- `dc.ds.knocktf()` — `.X` is already log2FCs; `obs["source"]` is the perturbed TF and
  `obs["logFC"]` its effect. It is the ground truth for `dc.bm.benchmark`, not a matrix
  you normalize. `thr_fc=-1` (default) keeps clearly down-regulated perturbations.
- `dc.ds.toy_bench()` — toy data with ground truth in `obs["source"]` / `class` /
  `type_p` for exercising the benchmark pipeline ([benchmarking.md](benchmarking.md)).

## Gene identifiers

Real datasets use gene **symbols** in `var_names` (matching the OmniPath priors). If you
bring data with Ensembl IDs, convert with `dc.ds.ensmbl_to_symbol(...)` so `net`
`target`s line up — mismatched namespaces get silently pruned by `tmin`
(see the identifier footgun in [priors.md](priors.md)).

## Related references

[calling-convention.md](calling-convention.md) (normalized-input requirement),
[preprocessing.md](preprocessing.md) (normalize / filter / pseudobulk / spatial kNN),
[priors.md](priors.md) (matching organism and gene identifiers),
[benchmarking.md](benchmarking.md) (`knocktf`, `toy_bench`).
