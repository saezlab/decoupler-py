# Contrast-level activity (bulk and pseudobulk)

"Which TFs or pathways are most active in condition A vs B" across replicate samples is answered
by running `ulm` on a **differential-expression statistic**, never by averaging per-sample or
per-cell scores. The documented route is: raw integer counts → gene filtering → PyDESeq2
Wald test → `ulm` on the `stat` column with CollecTRI, PROGENy or Hallmark.

## 1. Start from raw counts in an AnnData (samples × genes)

Bulk: counts are already per sample. Single-cell with donors: aggregate first.

```python
pdata = dc.pp.pseudobulk(adata=adata, sample_col="<sample or donor col>",
                         groups_col="<celltype col>", mode="sum")
```

`pseudobulk` requires raw integer counts in `.X` (or pass `layer=`); it refuses normalized data
unless `skip_checks=True`. Output obs gets `psbulk_cells` and `psbulk_counts`, and obs columns
that are constant within each sample×group (disease, sex, ...) are carried over. Then drop
low-quality pseudobulk samples with the defaults:

```python
dc.pp.filter_samples(pdata, min_cells=10, min_counts=1000)
```

Subset to the cell type of interest **before** gene filtering and DE:
`sub = pdata[pdata.obs["<celltype col>"] == "<type>"].copy()`.

## 2. Gene filtering (required; skipping it changes the answer)

```python
# edgeR-style filter; all keyword values shown are the defaults
dc.pp.filter_by_expr(adata=sub, group="<condition col>", min_count=10,
                     min_total_count=15, large_n=10, min_prop=0.7)
# pseudobulk only: keep genes expressed in a fraction of cells in enough samples
dc.pp.filter_by_prop(adata=sub, min_prop=0.1, min_smpls=2)
```

Both modify in place. Use `group=` the condition column. `dc.pl.filter_by_expr` and
`dc.pl.filter_by_prop` with the same arguments preview the thresholds.

If you normalize for PCA or QC, keep counts in a layer and swap them back before DESeq2:
`adata.layers["counts"] = adata.X.copy()` ... `dc.pp.swap_layer(adata, key="counts", inplace=True)`.

## 3. PyDESeq2 Wald statistic

```python
from pydeseq2.dds import DeseqDataSet, DefaultInference
from pydeseq2.ds import DeseqStats

inference = DefaultInference(n_cpus=8)
# bulk, two groups:            design="~<condition>"
# pseudobulk from single cell: design="~<condition> + <sample covariate>"  (e.g. "~disease + sex")
dds = DeseqDataSet(adata=sub, design="~<condition> + <covariate>", refit_cooks=True, inference=inference)
dds.deseq2()
stat_res = DeseqStats(dds, contrast=["<condition>", "<test level>", "<reference level>"],
                      inference=inference)
stat_res.summary()
results_df = stat_res.results_df           # columns: baseMean, log2FoldChange, lfcSE, stat, pvalue, padj
```

- Pass the AnnData directly (`adata=`); obs columns are the sample metadata.
- Design: in pseudobulk data include the sample-level covariates that exist in obs alongside the
  condition (the documented pseudobulk workflow adds the donor's sex to the disease term). Check
  `pdata.obs.columns` for such covariates; omitting an available one changes the ranking. For a
  plain two-group bulk design use `"~<condition>"` only.
- Contrast order is `[column, numerator, denominator]`; positive `stat` = up in the numerator.
- Make categorical obs columns plain strings if PyDESeq2 complains about dtype.

## 4. Score the statistic with ulm

```python
data = results_df[["stat"]].T.rename(index={"stat": "<test>.vs.<ref>"})  # 1 × genes DataFrame
net = dc.op.collectri(organism="human")       # or dc.op.progeny, dc.op.hallmark
acts, padj = dc.mt.ulm(data=data, net=net)    # DataFrames, 1 row × sources
ranked = acts.T.iloc[:, 0].sort_values(ascending=False)   # most activated first
```

Use `stat` (the Wald t-value), not `log2FoldChange`: it carries both effect size and
confidence. The vignette filters to `padj < 0.05` only for plotting (`msk = (padj.T < 0.05).iloc[:, 0]`);
for a ranking question rank by the score. No normalization is needed for contrast statistics.

Plots: `dc.pl.barplot(data=acts, name="<row name>", top=25)`,
`dc.pl.volcano(data=results_df, x="log2FoldChange", y="padj", net=net, name="<source>")`,
`dc.pl.source_targets(data=results_df, x="weight", y="stat", net=net, name="<source>")`,
`dc.pl.leading_edge(results_df, stat="stat", net=net, name="<source>")`.
