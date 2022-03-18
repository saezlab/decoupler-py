Release notes
=============

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
