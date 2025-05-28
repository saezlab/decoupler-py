# decoupler - Ensemble of methods to infer enrichment scores
<img src="https://github.com/saezlab/decoupleR/blob/master/inst/figures/logo.svg?raw=1" align="right" width="120" class="no-scaled-link" />
   

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[![Issues][badge-issues]][issue tracker]
[![Coverage][badge-coverage]][codecoverage]
[![Stars][badge-stars]](https://github.com/scverse/anndata/stargazers)

[![PyPI][badge-pypi]][pypi]
[![Downloads month][badge-mdown]][down]
[![Downloads all][badge-adown]][down]

[![Conda version][badge-condav]][conda]
[![Conda downloads][badge-condad]][conda]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/saezlab/decoupler-py/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/decoupler-py
[badge-condav]: https://img.shields.io/conda/vn/conda-forge/decoupler-py.svg
[badge-condad]: https://img.shields.io/conda/dn/conda-forge/decoupler-py.svg
[badge-issues]: https://img.shields.io/github/issues/saezlab/decoupler-py
[badge-coverage]: https://codecov.io/gh/saezlab/decoupler-py/branch/main/graph/badge.svg
[badge-pypi]: https://img.shields.io/pypi/v/decoupler.svg
[badge-mdown]: https://static.pepy.tech/badge/decoupler/month
[badge-adown]: https://static.pepy.tech/badge/decoupler
[badge-stars]: https://img.shields.io/github/stars/saezlab/decoupler-py?style=flat&logo=github&color=yellow

`decoupler` is a python package containing different enrichment statistical
methods to extract biologically driven scores
from omics data within a unified framework. This is its faster and memory efficient Python implementation,
a deprecated version in R can be found [here](https://github.com/saezlab/decoupler).

It is a package from the [scverse][] ecosystem {cite:p}`scverse`,
designed for easy interoperability with `anndata`, `scanpy` {cite:p}`scanpy` and other related packages.

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

## Installation

You need to have Python 3.10 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

There are several alternative options to install decoupler:

1. Install the latest stable release from [PyPI][pypi] with minimal dependancies:

```bash
pip install decoupler
```

2. Install the latest stable full release from [PyPI][pypi] with extra dependancies:

```bash
pip install decoupler[full]
```

3. Install the latest stable version from [conda-forge][conda] using mamba or conda (pay attention to the `-py` suffix at the end):

```bash
mamba create -n=dcp conda-forge::decoupler-py
```

4. Install the latest development version:

```bash
pip install git+https://github.com/saezlab/decoupler-py.git@main
```

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

> Badia-i-Mompel P., Vélez Santiago J., Braunger J., Geiss C., Dimitrov D.,
Müller-Dott S., Taus P., Dugourd A., Holland C.H., Ramirez Flores R.O.
and Saez-Rodriguez J. 2022. decoupleR: Ensemble of computational methods
to infer biological activities from omics data. Bioinformatics Advances.
<https://doi.org/10.1093/bioadv/vbac016>

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[scverse]: https://scverse.org/
[issue tracker]: https://github.com/saezlab/decoupler-py/issues
[tests]: https://github.com/saezlab/decoupler-py/actions/workflows/test.yaml
[documentation]: https://decoupler-py.readthedocs.io
[changelog]: https://decoupler-py.readthedocs.io/en/latest/changelog.html
[api documentation]: https://decoupler-py.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/decoupler
[down]: https://pepy.tech/project/decoupler
[conda]: https://anaconda.org/conda-forge/decoupler-py
[codecoverage]: https://codecov.io/gh/saezlab/decoupler-py
