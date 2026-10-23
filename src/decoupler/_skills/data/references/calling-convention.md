# The unified method call

Every enrichment method in `dc.mt` is a callable object with the **same signature**.
Learn it once and it applies to `ulm`, `mlm`, `ora`, `gsea`, `gsva`, `aucell`,
`viper`, `zscore`, `waggr`, `mdt`, `udt` (and `dc.mt.decouple`).

```python
import decoupler as dc

adata, net = dc.ds.toy()          # AnnData (obs × var) + long-format net
dc.mt.ulm(adata, net, tmin=3)     # writes adata.obsm["score_ulm"], ["padj_ulm"]
```

## Shared signature

```python
dc.mt.<method>(
    data,              # AnnData | pandas.DataFrame | [matrix, obs_names, var_names]
    net,               # long-format DataFrame: source, target, [weight]
    tmin=5,            # drop sources with < tmin targets present in the data
    raw=False,         # use adata.raw instead of adata.X
    layer=None,        # use adata.layers[layer] instead of adata.X (passed via kwargs)
    empty=True,        # drop all-zero observations/features before scoring
    bsize=250_000,     # batch size (rows) for sparse / backed matrices
    verbose=False,
    **kwargs,          # method-specific args — see references/methods.md
)
```

`layer=` is accepted even though it is not in the explicit signature — it is forwarded
through `**kwargs` to the runner. Use `layer=` **or** `raw=`, not both.

## Accepted input types (`data`)

| Form | What it is | Returns |
|---|---|---|
| `AnnData` | scored on `.X` (or `.raw`/`layer`) | **`None`** — results written to `.obsm` in place |
| `pandas.DataFrame` | observations (rows) × features (cols) | `(es, pv)` tuple of DataFrames |
| `list` `[matrix, obs_names, var_names]` | numpy/sparse matrix + name arrays | `(es, pv)` tuple of DataFrames |

See [io-and-outputs.md](io-and-outputs.md) for the return details.

### Footgun — the matrix form must be a `list`, not a `tuple`

```python
# WRONG — a tuple raises "mat must be a list of [matrix, samples, features]"
es, pv = dc.mt.ulm((X, obs, var), net)

# RIGHT — wrap it in a list
es, pv = dc.mt.ulm([X, obs, var], net)
```

## The three footguns that silently give wrong results

1. **Input must be normalized, not raw counts.** decoupler expects
   log-normalized expression — the same matrix you would use for differential
   expression. **When normalizing, use `sc.pp.normalize_total(adata, target_sum=1e4)`**
   followed by `sc.pp.log1p(adata)`. Passing raw counts runs without error but produces
   meaningless scores. If your normalized data is in a layer, pass
   `layer="lognorm"`; if in `.raw`, pass `raw=True`. **Check before normalizing:** if
   `.X` is integer-valued with a large max it is raw counts; if it is small floats
   (often with `adata.uns["log1p"]`) it is already normalized — do not normalize twice.
   The bundled datasets differ; see [datasets.md](datasets.md) (e.g. `pbmc3k` is already
   log-normalized, `covid5k` is raw counts).

2. **`net` must be long-format** with columns `source`, `target`, and (for weighted
   methods) `weight` — one row per (program, feature) edge. A wide/matrix net or wrong
   column names will fail or misbehave. See [priors.md](priors.md).

3. **`tmin` silently prunes.** Sources with fewer than `tmin` (default 5) targets
   present in `data` are dropped *before* scoring — they simply won't appear in the
   output. On small or toy nets this can drop everything; the examples use `tmin=3`.
   Run with `verbose=True` to see `<method> - X sources with < tmin targets` and how
   many survived.

## Listing and introspecting methods

```python
dc.mt.show()          # DataFrame: name, desc, stype, weight, test, limits, reference
```

- `weight` — whether the method uses signed edge weights (`True` for ulm/mlm/viper/…).
- `test` — whether it produces p-values (`padj`). `False` for aucell/gsva/mdt/udt.
- `stype` / `limits` — value type and output range (e.g. aucell `(0, 1)`).

## Related references

[io-and-outputs.md](io-and-outputs.md) (return types, reading `.obsm` back),
[methods.md](methods.md) (per-method kwargs and how to choose one),
[priors.md](priors.md) (building/fetching `net`),
[preprocessing.md](preprocessing.md) (getting a normalized matrix / layers).
