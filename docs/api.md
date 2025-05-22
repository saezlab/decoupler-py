# API

## Datasets

### Bulk
```{eval-rst}
.. module:: decoupler.ds
.. currentmodule:: decoupler

.. autosummary::
    :toctree: generated

    ds.hsctgfb
    ds.knocktf
```

### Single-cell
```{eval-rst}
.. autosummary::
    :toctree: generated

    ds.covid5k
    ds.erygast1k
    ds.pbmc3k
```

### Spatial
```{eval-rst}
.. autosummary::
    :toctree: generated

    ds.msvisium
```

### Toy
```{eval-rst}
.. autosummary::
    :toctree: generated

    ds.toy
    ds.toy_bench
```

### Utils
```{eval-rst}
.. autosummary::
    :toctree: generated

    ds.ensmbl_to_symbol
```

## Methods
### Single methods
```{eval-rst}
.. module:: decoupler.mt
.. currentmodule:: decoupler

.. autosummary::
    :toctree: generated

    mt.aucell
    mt.gsea
    mt.gsva
    mt.mdt
    mt.mlm
    mt.ora
    mt.udt
    mt.udt
    mt.ulm
    mt.viper
    mt.waggr
    mt.zscore
```

### Multiple methods
```{eval-rst}
.. autosummary::
    :toctree: generated

    mt.decouple
    mt.consensus
```

## OmniPath

### Resources
```{eval-rst}
.. module:: decoupler.op
.. currentmodule:: decoupler

.. autosummary::
    :toctree: generated

    op.collectri
    op.dorothea
    op.hallmark
    op.progeny
    op.resource
```

### Utils
```{eval-rst}
.. autosummary::
    :toctree: generated

    op.show_resources
    op.show_organisms
    op.translate
```

## Plotting

```{eval-rst}
.. module:: decoupler.pl
.. currentmodule:: decoupler

.. autosummary::
    :toctree: generated

    pl.barplot
    pl.dotplot
    pl.filter_by_expr
    pl.filter_by_prop
    pl.filter_samples
    pl.leading_edge
    pl.network
    pl.obsbar
    pl.obsm
    pl.order_targets
    pl.order
    pl.source_targets
    pl.volcano
```

## Preprocessing

### Data
```{eval-rst}
.. module:: decoupler.pp
.. currentmodule:: decoupler

.. autosummary::
    :toctree: generated

    pp.extract
```

### Network
```{eval-rst}
.. autosummary::
    :toctree: generated

    pp.read_gmt
    pp.prune
    pp.adjmat
    pp.idxmat
    pp.shuffle_net
    pp.net_corr
```

### AnnData
```{eval-rst}
.. autosummary::
    :toctree: generated

    pp.get_obsm
    pp.swap_layer
    pp.pseudobulk
    pp.filter_samples
    pp.filter_by_expr
    pp.filter_by_prop
    pp.knn
    pp.bin_order
```

## Tools

```{eval-rst}
.. module:: decoupler.tl
.. currentmodule:: decoupler

.. autosummary::
    :toctree: generated

    tl.rankby_group
    tl.rankby_obsm
    tl.rankby_order
```
