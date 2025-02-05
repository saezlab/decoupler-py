# decoupler - Ensemble of methods to infer biological activities <img src="https://github.com/saezlab/decoupleR/blob/master/inst/figures/logo.svg?raw=1" align="right" width="120" class="no-scaled-link" />
<!-- badges: start -->
[![main](https://github.com/saezlab/decoupler-py/actions/workflows/ci.yml/badge.svg)](https://github.com/saezlab/decoupler-py/actions)
[![GitHub issues](https://img.shields.io/github/issues/saezlab/decoupler-py.svg)](https://github.com/saezlab/decoupler-py/issues/)
[![Documentation Status](https://readthedocs.org/projects/decoupler-py/badge/?version=latest)](https://decoupler-py.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/saezlab/decoupler-py/branch/main/graph/badge.svg?token=TM0P29KKN5)](https://codecov.io/gh/saezlab/decoupler-py)
[![Downloads](https://static.pepy.tech/badge/decoupler)](https://pepy.tech/project/decoupler)

[![Conda Recipe](https://img.shields.io/badge/recipe-decoupler--py-green.svg)](https://anaconda.org/conda-forge/decoupler-py)
[![Conda page](https://img.shields.io/conda/vn/conda-forge/decoupler-py.svg)](https://anaconda.org/conda-forge/decoupler-py)
[![Conda downloads](https://img.shields.io/conda/dn/conda-forge/decoupler-py.svg)](https://anaconda.org/conda-forge/decoupler-py)
<!-- badges: end -->

`decoupler` is a package containing different enrichment statistical methods to extract biologically driven scores from omics data within a unified framework.
This is its faster and memory efficient Python implementation, for the R version go [here](https://github.com/saezlab/decoupleR).

For further information and example tutorials, please check our [documentation](https://decoupler-py.readthedocs.io/en/latest/index.html).

If you have any question or problem do not hesitate to open an [issue](https://github.com/saezlab/decoupler-py/issues).

## Installation

`decoupler` can be installed from `pip` (lightweight installation)::
```
pip install decoupler
```

It can also be installed from `conda` and `mamba` (this includes extra dependencies):
```
mamba create -n=decoupler conda-forge::decoupler-py
```

Alternatively, to stay up-to-date with the newest unreleased version, install from source: 
```
pip install git+https://github.com/saezlab/decoupler-py.git
```

## scverse
`decoupler` is part of the [scverse](https://scverse.org) ecosystem, a collection of tools for single-cell omics data analysis in python.
For more information check the link.

## License
Enrichment methods inside decoupler can be used for academic or commercial purposes, except `viper` which holds a non-commercial license. 

The data redistributed by OmniPath does not have a single license, each original resource has its own. By default, `decoupler`
assumes an academic license, but commercial or nonprofit licenses can be specified in the `license` parameter of `decoupler`'s OmniPath functions.
[Here](https://omnipathdb.org/info) one can find the license information of all the resources in OmniPath.

## Citation

Badia-i-Mompel P., Vélez Santiago J., Braunger J., Geiss C., Dimitrov D.,
Müller-Dott S., Taus P., Dugourd A., Holland C.H., Ramirez Flores R.O.
and Saez-Rodriguez J. 2022. decoupleR: Ensemble of computational methods
to infer biological activities from omics data. Bioinformatics Advances.
<https://doi.org/10.1093/bioadv/vbac016>
