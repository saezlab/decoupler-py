# decoupler - Ensemble of methods to infer enrichment scores
<img src="https://raw.githubusercontent.com/scverse/decoupler/refs/heads/main/docs/_static/images/logo.svg" align="right" width="120" class="no-scaled-link" />

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

[badge-tests]: https://img.shields.io/github/actions/workflow/status/scverse/decoupler/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/decoupler
[badge-condav]: https://img.shields.io/conda/vn/conda-forge/decoupler-py.svg
[badge-condad]: https://img.shields.io/conda/dn/conda-forge/decoupler-py.svg
[badge-issues]: https://img.shields.io/github/issues/scverse/decoupler
[badge-coverage]: https://codecov.io/gh/scverse/decoupler/branch/main/graph/badge.svg
[badge-pypi]: https://img.shields.io/pypi/v/decoupler.svg
[badge-mdown]: https://static.pepy.tech/badge/decoupler/month
[badge-adown]: https://static.pepy.tech/badge/decoupler
[badge-stars]: https://img.shields.io/github/stars/scverse/decoupler?style=flat&logo=github&color=yellow

`decoupler` is a python package containing different enrichment statistical methods to extract biologically driven scores from omics data within a unified framework.
This is its faster and memory efficient Python implementation, a deprecated version in R can be found [here](https://github.com/saezlab/decoupler).

[//]: # (numfocus-fiscal-sponsor-attribution)

decoupler is part of the scverse® project ([website](https://scverse.org), [governance](https://scverse.org/about/roles)) and is fiscally sponsored by [NumFOCUS](https://numfocus.org/).
If you like scverse® and want to support our mission, please consider making a tax-deductible [donation](https://numfocus.org/donate-to-scverse) to help the project pay for developer time, professional services, travel, workshops, and a variety of other needs.

<div align="center">
<a href="https://numfocus.org/project/scverse">
  <img
    src="https://raw.githubusercontent.com/numfocus/templates/master/images/numfocus-logo.png"
    width="200"
  >
</a>
</div>

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

## Installation

You need to have Python 3.11 or newer installed on your system.
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
pip install git+https://github.com/scverse/decoupler.git@main
```

## Claude Code Skill

decoupler ships an [Agent Skill](https://docs.claude.com/en/docs/claude-code/skills) for [Claude Code](https://claude.com/claude-code) that teaches the agent to use the library correctly. Install it once into your personal skills directory:

```bash
decoupler-install-skills
```

This copies the skill to `~/.claude/skills/decoupler/`, making it available in **every** project (re-run with `--force` after upgrading). Once installed, just ask Claude Code to do decoupler tasks ("find the most active TFs in this cell cluster", "score hallmark gene sets with ora", "build a consensus across methods") and the skill is consulted automatically. To avoid copying into your home directory, point Claude Code at the bundled copy instead:

```bash
export CLAUDE_SKILLS_PATH="$(decoupler-install-skills --print-path)"
```

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## License
Enrichment methods inside decoupler can be used for academic or commercial purposes, except `viper` which holds a non-commercial license.

The data redistributed by OmniPath does not have a single license, each original resource has its own. By default, `decoupler`
assumes an academic license, but commercial or nonprofit licenses can be specified in the `license` parameter of `decoupler`'s OmniPath functions.
[Here](https://omnipathdb.org/info) one can find the license information of all the resources in OmniPath.

## Citation

> Badia-i-Mompel P., Vélez Santiago J., Braunger J., Geiss C., Dimitrov D.,
Müller-Dott S., Taus P., Dugourd A., Holland C.H., Ramirez Flores R.O.
and Saez-Rodriguez J. 2022. decoupleR: Ensemble of computational methods
to infer biological activities from omics data. Bioinformatics Advances.
<https://doi.org/10.1093/bioadv/vbac016>

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[scverse]: https://scverse.org/
[issue tracker]: https://github.com/scverse/decoupler/issues
[tests]: https://github.com/scverse/decoupler/actions/workflows/test.yaml
[documentation]: https://decoupler.scverse.org/
[changelog]: https://decoupler.scverse.org/page/changelog.html
[api documentation]: https://decoupler.scverse.org/page/api.html
[pypi]: https://pypi.org/project/decoupler
[down]: https://pepy.tech/project/decoupler
[conda]: https://anaconda.org/conda-forge/decoupler-py
[codecoverage]: https://codecov.io/gh/scverse/decoupler
