# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## 2.2.1

### Changes
- `pp.filter_by_prop` now defaults to `min_prop=0.1` (was `0.2`) to match the value used in the vignettes
- `ds.msvisium` now returns log-normalized data (`sc.pp.normalize_total` + `sc.pp.log1p`), with the log-normalized matrix also stored in `adata.layers["norm"]`, instead of raw counts

### Bugfixes
- Fixed `mt.ora` selecting the bottom `nvar - n_up` features instead of the top `n_up` ones. Ranks are ascending, so the threshold is now `nvar - n_up`, restoring the behaviour of `decoupler<2`. With the default `n_up` (top 5%) this selected 95% of the features as observed, which also made `mt.ora` fail with `invalid contingency table` whenever `nvar - n_up` exceeded `n_bg` (#346)
- Fixed `n_bm` in `mt.ora` selecting the bottom `n_bm - 1` features instead of `n_bm`
- `mt.ora` now validates that `n_up` and `n_bm` do not overlap and that `n_bg` is large enough for the number of selected features, instead of failing inside the Fisher exact test
- `mt.query_set` now raises an informative error when `n_bg` is smaller than the number of features to test, instead of failing inside `scipy.stats.fisher_exact`
- Fixed `mt.gsea` not shuffling the permutation indices when `seed=0`, which produced identical permutations and invalid p-values (#355)
- Fixed `mt.waggr` folding its already two-sided permutation p-values as one-sided, which gave scores near 0 the smallest p-value (#373)

## 2.2.0

### Added
- Bundled a Claude Code [Agent Skill](https://docs.claude.com/en/docs/claude-code/skills) that teaches coding agents to use decoupler correctly (method selection, calling conventions, output locations, priors, datasets, and benchmarking)
- Added the `decoupler-install-skills` console script to install the skill into `~/.claude/skills/decoupler/` (use `--force` to refresh after upgrading, or `--print-path` to point `CLAUDE_SKILLS_PATH` at the bundled copy in place)

## 2.1.6

### Bugfixes
- Fixed `op.translate` (and every `op.resource` call for non-human organisms) failing with HTTP 404 after HGNC dropped the `genenames/` subtree from the EBI FTP mirror; the HCOP fifteen-column files are now fetched from the new HGNC Google Cloud Storage bucket (#303)
- Removed a redundant `pd.read_csv` call in `op.translate` that re-downloaded the HCOP file a second time and bypassed the retry logic in `_download`
- Fixed `bm.benchmark` ground-truth matrix row order not matching score matrix after `pivot`, which silently misaligned perturbation labels when obs names sort differently from their original order (#302)

## 2.1.5

### Added
- `pl.volcano` now accepts a gene name (`str`) or list of gene names (`list[str]`) for the `top` parameter to annotate specific features on volcano plots

### Changes
- Refactored `ds.ensmbl_to_symbol` to reuse `_download` and fixed mirror fallback to actually switch between Ensembl mirrors
- Made `tqdm` progress bar compact in `_download`

### Bugfixes
- Fixed `pl.dotplot` ignoring `vcenter=0` due to falsy check (#293)
- Fixed `_log` setting the root logging level to INFO for all packages (#296)

## 2.1.4

### Changes
- Added `alternative` argument to `decoupler.mt.query_set`. By default `'greater'`, before it was `'two-sided'`
- Unpinned `scipy` version limit

### Bugfixes
- Fixed missing progressbar for `decoupler._download._download`
- Added missing `decoupler.mt.query_set` documentation

## 2.1.3

### Changes
- `pp.adjmat` now returns the same features as used as input instead of the subset of `net`
- `pp.pseudobulk` now returns the same order features as used as input instead of shuffling them
- Added a dedicated header and 5 attemps to `_download` to mitigate 429 Client Error from Zenodo downloads

## 2.1.2

### Added
- `pp.query_set` to test overlap between a given feature set against a database of sets

### Changes

- `tl.rankby_obsm` now accepts `AnnData.obs` column names specified in the `obs_keys` argument
- Most plotting functions now accept extra arguments through `kw_` arguments.
- p-values now are corrected using a custom numba-optimized version of `scipy.stats.false_discovery_control` called `_fdr_bh_axis1_numba`

## 2.1.1

### Added
- mypy checks in CI
- notebook checks in CI

### Bugfixes
- Fixed error in `decoupler.pp.pseudobulk` when `adata.obs_names` were not unique, now throws verbose error
- Fixed corner case in `decoupler.mt.gsea` when p-values were infitite and could not be corrected

## 2.1.0

### Changes

- Updated logo
- `decoupler._download._download` now returns bytes instead of a dataframe. To transform to `pandas.DataFrame` use `decoupler._download._bytes_to_pandas`
- Enrichment methods and pseudobulking now work with backed AnnData objects, useful when working with big datasets and memory is limited

### Bugfixes
- Fixed error in `pl.obsm` where default value of `cmap_obs` was not properly set.

## 2.0.7

### Changes
- Added `pre-commit` functionality to the repository

## 2.0.6

### Changes
- Modified links and paths to follow scverse's repository

## 2.0.5

### Bugfixes
- Fixed error message when extra dependencies where not installed

## 2.0.4

### Bugfixes
- Fixed `dcor` import bug as an external dependency

## 2.0.3

### Bugfixes
- Fixed error in `pp.pseudobulk` when obs columns were not categorical

## 2.0.2

### Changes
- Allowed ordering functions (`pp.bin_order`, `pl.order` and `pl.order_targets`) not to be bound between 0 and 1
- Added ipywidgets as dependency
- Silenced xgboost warnings

### Bugfixes
- Handled corner case in `bm.metric.auc` when scores are all 0
- Fixed error in `bm.metric.hmean` when metrics were str instead of list
- Fixed error when `obs` column is a list in `pp.pseudobulk`

## 2.0.1

### Bugfixes
- Fixed an error in `pp.pseudobulk` when handling empty samples or features

## 2.0.0

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
