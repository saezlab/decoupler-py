Release notes
=============

1.2.0
-----

Bug fixes
~~~~~~~~~
- Removed ``python <3.10`` limitation.
- Forced ``np.float32`` type to output of ``get_contrast``.

Changes
~~~~~~~~
- ``extract`` now removes empty samples and features.
- ``run_consensus`` now follows the same format as other methods, old function is now called ``cons``.
- ``get_pseudobulk`` now checks if input are raw integer counts.
- ``plot_volcano`` now can also plot without subsetting features by a network and can save plots to disk.
- ``get_top_targets`` now can also work without subsetting features by a network and returns significant adjusted p-values.
- ``get_contrast`` now can also work without needing to group.

Additions
~~~~~~~~~
- Added ``dense_run`` util function which runs methods ignoring zeros in the data.
- Added ``plot_violins`` and ``plot_barplot`` functions.
- Added ``p_adjust_fdr`` util function to correct p-values for FDR.

1.1.0
-----
Bug fixes
~~~~~~~~~
- Fixed ``get_pseudobulk`` errors.
- Fixed ``get_progeny`` to correctly return non duplicate entries.
- Fixed ``run_viper`` parallelization error.
- Fixed ``run_ora`` to correctly deal with random ties.

Changes
~~~~~~~~
- ``get_dorothea`` now returns an ordered dataframe.
- ``get_contrast`` now prints warnings instead of returning an empty dataframe.

Additions
~~~~~~~~~
- Added ``get_top_targets`` util function.
- Added ``format_contrast_results`` util function.
