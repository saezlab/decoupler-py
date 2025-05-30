# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## 2.0.3

Major update to accomodate the scverse template {cite}`scverse`.

All functions have been rewritten to follow the new API, errors when running previous versions (`1.X.X`) are expected if `decoupler >= 2.0.0` is installed.

### Changes

- Methods are now in the `mt` module and are built from shared class `Method`
  - Use `decoupler.mt.<method_name>` to call a method
  - `min_n` argument has been renamed `tmin`
  - New argument `bsize` allows to run a method with batches in case excessive memory usage is an issue
  - $p_{values}$ of the enrichment scores are now corrected by Benjamini-Hochberg
  - `mdt` and `udt` are now based on `xgboost` instead of `sklearn` for better scalability. `udt` statistic is now the coefficient of determination $R^2$ instead of the importance of a single decision tree.
  - `mlm` and `ulm` now include a `tval` parameter, which allows returning either the t-value of the slope or the slope itself as the enrichment statistic
  - `ora` now returns the odds ratio of the contingency table as a statistic, and computes a two-sided Fisher exact test instead of a one-sided one
  - `viper` now correctly estimates shadow regulons when network weights are values other than -1 or +1
  - `wsum` and `wmean` are deprecated, instead now the method `waggr` allows to run both methods and any custom function. This makes it easier to quickly test new enrichment methods without having to deal with `decoupler`'s implementation
- Databases from Omnipath can now be accessed through the new `op` module
  - Use `decoupler.op.<resource_name>` to access a database  
  - Removed the `omnipath` package as a dependancy
  - Fixed `collectri` to the publication version instead of the OmniPath one
  - Made `progeny` only return significant genes by default instead of the top N genes per pathway
- Plots are now in a new `pl` module
    - Use `decoupler.pl.<plot_name>` to call a plot
    - They use a common class `Plotter` to make it easier to share arguments between them
    - `plot_violins` has been deprecated
    - Names that have changed
      - `plot_psbulk_samples` to `filter_samples`
      - `plot_running_score` to `leading_edge`
      - `plot_associations` to `obsm`
      - `plot_targets` to `source_targets`
- Preprocessing functions are now in the new `pp` module
  - Renamed `check_corr` to `net_corr`, now also returns adjusted $p_{values}$
  - Renamed `get_acts` to `get_obsm`
  - Renamed `get_pseudobulk` to `pseudobulk`. Now it does not automatically remove low quality samples, this is now done with the function `filter_samples`
  - Deprecated `get_contrast`, `get_top_targets` and `format_contrast_results`. `PyDESeq2` should be used instead
  - Moved `rank_sources_groups` to the new `tl` module as `rankby_group`
  - Moved `get_metadata_associations` to the new `tl` module as `rankby_obsm`
- Moved the benchmarking pipeline inside a new `bm` module, with its metrics and plotting functions in further submodules (`bm.metric` and `bm.pl`)

### Added

- `ds` module with functions to download several datasets at different resolutions
    - Bulk: `hsctgfb` and `knocktf`
    - Single-cell: `pbmc3k`, `covid5k` and `erygast1k`
    - Spatial: `msvisium`
    - Toy data: `toy` and `toy_bench`
    - Utils: `ensmbl_to_symbol`
- New database functions in the `op` module
  - Added `hallmark` as a custom resource
- New plotting funcitons in the `pl` module
  - Added `obsbar` to plot size of metadata columns in `anndata.AnnData.obs`
  - Added `order` to plot sources or features along a continous process such as a trajectory
  - Added `order_targets` to plot the targets of a given source along a continous process
  - 
- New preprocessing functions in the `pp` module
  - Added two functions to format networks, `adjmat` to return an adjacency matrix, and `idxmax` to return a list of sets
  - Added `filter_samples` to filter pseudobulk profiles after running `pseudobulk`
  - Added `knn` to calculate K-Nearest Neighbors similarities based on spatial distances
  - Added `bin_order` to bin features across a continous process
- `tl` module with functions to perform statistical tests
  - Added `rankby_order` to test for non-linear associations of features with a continous process
- New benchmarking metrics and plotting related functions in the `bm` module
  - Added two more metrics, `F-score` and `qrank`
  - Added shared plots for metrics, `bm.pl.auc`, `bm.pl.fscore` and `bm.pl.qrank`
  - Added a summary plot across metrics `bm.pl.summary`
