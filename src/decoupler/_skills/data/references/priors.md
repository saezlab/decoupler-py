# Prior knowledge (the `net`)

Every method needs a **`net`**: a long-format `pandas.DataFrame` describing which
features (genes) belong to which biological program.

## Net format

| column | meaning | required |
|---|---|---|
| `source` | the program (TF, pathway, gene set) | yes |
| `target` | the feature (gene) | yes |
| `weight` | signed importance of the edge | only for weighted methods |

One row per edge. Weighted methods (`ulm`, `mlm`, `viper`, `zscore`, `waggr`, `mdt`,
`udt`) use `weight` (sign encodes activation/repression); gene-set methods (`ora`,
`gsea`, `aucell`, `gsva`) ignore it.

```python
import decoupler as dc
net = dc.op.collectri(organism="human")
net.head()      # columns: source, target, weight
```

## Fetching curated priors from `dc.op` (OmniPath)

All default to `organism="human"` and `license="academic"`.

- **TF regulons**
  - `dc.op.collectri(organism="human", remove_complexes=False)` — CollecTRI TF→gene
    regulons with signed weights. Recommended default for TF activity.
  - `dc.op.dorothea(organism="human", levels=None)` — DoRothEA regulons; `levels`
    selects confidence levels (e.g. `["A", "B", "C"]`).
- **Pathways**
  - `dc.op.progeny(organism="human", top=inf, thr_padj=0.05)` — PROGENy responsive
    genes; `top` limits genes per pathway (e.g. `top=500` is common).
- **Gene sets**
  - `dc.op.hallmark(organism="human")` — MSigDB hallmark gene sets (no weights → use
    with `ora`/`gsea`/`aucell`/`gsva`).
- **Anything else on OmniPath**
  - `dc.op.resource(name, organism="human")` — fetch an arbitrary resource by name.
  - `dc.op.show_resources()` — list available resource names.

## Loading your own gene sets from a `.gmt`

```python
net = dc.pp.read_gmt("my_sets.gmt")   # -> long-format source/target (no weights)
```

Use with a gene-set method (`ora`, `gsea`, `aucell`, `gsva`).

## Cross-species: translate a net to another organism

`dc.op` priors accept `organism=` directly; to translate an existing human net to
another species by orthology use `dc.op.translate`:

```python
mouse_net = dc.op.translate(net, target_organism="mouse", min_evidence=3)
dc.op.show_organisms()   # list supported organisms
```

- Translates the `source`/`target` gene symbols (control columns via `columns=`).
- `min_evidence` and `one_to_many` control ortholog stringency.

## Footguns

- **Match gene identifiers.** The net's `target` symbols must match your data's
  `var_names` (both symbols, or both Ensembl IDs). Mismatched namespaces yield empty
  results after `tmin` pruning. `dc.ds.ensmbl_to_symbol` helps convert.
- **Right net for the method.** Weighted methods on an unweighted net treat all
  weights as 1 (loses activation/repression sign); gene-set methods on a weighted net
  ignore the weights. Pick the pair deliberately — see [methods.md](methods.md).
- **`organism` must match your data.** Don't score mouse data with a human net;
  fetch with `organism="mouse"` or `dc.op.translate`.

## Related references

[methods.md](methods.md) (weighted vs gene-set methods),
[calling-convention.md](calling-convention.md) (how `net` is consumed, `tmin`),
[getting-started.md](getting-started.md) (end-to-end example).
