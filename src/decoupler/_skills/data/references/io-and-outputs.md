# Where the results go (input type decides the return)

The **return type of every `dc.mt` method depends on the input type**. This is the
single most common source of confusion — an AnnData call looks like it returned
nothing, but it wrote the results into the object.

## AnnData in → results written in place, returns `None`

```python
adata, net = dc.ds.toy()
out = dc.mt.ulm(adata, net, tmin=3)
assert out is None                       # nothing is returned

adata.obsm["score_ulm"]                  # DataFrame: obs × sources (activity scores)
adata.obsm["padj_ulm"]                   # DataFrame: obs × sources (FDR-adj p-values)
```

- Results are stored under `.obsm` with keys `score_<method>` and (only if the method
  tests) `padj_<method>`. Method names: `score_ulm`, `padj_ulm`, `score_ora`, etc.
- **Do not** write `adata = dc.mt.ulm(adata, net)` — that overwrites `adata` with
  `None` and loses your data.
- Non-testing methods (`aucell`, `gsva`, `mdt`, `udt`) write only `score_<method>`,
  no `padj_<method>`.
- Edge case: if `empty=True` dropped observations, decoupler returns a repaired copy
  of the AnnData instead of `None`; capture it if you pass data with empty rows.

### Reading scores back out as an AnnData

`dc.pp.get_obsm` pulls an `.obsm` score matrix into a new AnnData whose `.X` is the
scores and whose `var` are the sources. It **preserves `obs`, `uns`, and `obsm`** from
the original, so the result is ready for scanpy — you can `sc.pl.umap` / `sc.pl.spatial`
it directly, coloring by a source (`var_name`) or an `obs` column. This is the intended
way to visualize scores; see
[ranking-and-plotting.md](ranking-and-plotting.md).

```python
acts = dc.pp.get_obsm(adata, key="score_ulm")   # AnnData: obs × sources
acts.var_names                                   # the programs (e.g. TFs)
# acts.obs / acts.uns / acts.obsm are carried over from adata (e.g. spatial images)
```

## DataFrame or list in → returns an `(es, pv)` tuple

```python
import pandas as pd
df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)

es, pv = dc.mt.ulm(df, net, tmin=3)   # es: obs × sources scores; pv: obs × sources padj
```

- `es` (enrichment scores) and `pv` (adjusted p-values) are both `obs × sources`
  DataFrames.
- For **non-testing** methods, `pv is None`:
  ```python
  es, pv = dc.mt.aucell(df, net, tmin=3)
  assert pv is None
  ```
- The `list` form `[matrix, obs_names, var_names]` returns the same `(es, pv)` tuple.

## Multi-method returns

- `dc.mt.decouple(...)` returns a **dict** keyed `score_<m>` / `padj_<m>` for DataFrame
  input, or writes those keys into `.obsm` for AnnData input (returns `None`).
- `dc.mt.consensus(...)` reads all `score_*` entries and returns/writes a single
  consensus score. See [multi-method.md](multi-method.md).

## Quick decision table

| You passed… | You get back… | Where scores are |
|---|---|---|
| `AnnData` | `None` | `adata.obsm["score_<m>"]`, `["padj_<m>"]` |
| `DataFrame` / `list` | `(es, pv)` | the returned DataFrames (`pv=None` if no test) |
| `decouple(AnnData)` | `None` | multiple `adata.obsm[...]` keys |
| `decouple(DataFrame)` | `dict` | `dict["score_<m>"]`, `dict["padj_<m>"]` |

## Related references

[calling-convention.md](calling-convention.md) (input types and shared args),
[ranking-and-plotting.md](ranking-and-plotting.md) (using the scores downstream),
[multi-method.md](multi-method.md) (`decouple` / `consensus` outputs).
