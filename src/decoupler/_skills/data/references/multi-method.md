# Running many methods + consensus

When you don't want to commit to one method, run several and combine them. Two tools:
`dc.mt.decouple` (run N methods) and `dc.mt.consensus` (aggregate their scores).

## `dc.mt.decouple` — run several methods at once

```python
dc.mt.decouple(
    data, net,
    methods="all",     # or a list, e.g. ["ulm", "ora", "aucell"]
    args=None,         # per-method kwargs: {"ora": {"n_up": 300}, "gsea": {"seed": 0}}
    cons=False,        # also compute a consensus score
    **kwargs,          # shared kwargs forwarded to every method (e.g. tmin=3)
)
```

- **AnnData input** → writes `score_<m>` / `padj_<m>` for each method into `.obsm`,
  returns `None`.
- **DataFrame input** → returns a **dict** keyed `score_<m>` / `padj_<m>`.
- `methods="all"` runs every applicable method; pass a list to restrict.
- `args` gives per-method overrides; top-level `**kwargs` (like `tmin`) apply to all.

```python
adata, net = dc.ds.toy()
dc.mt.decouple(adata, net, methods=["ulm", "ora"], tmin=3)
adata.obsm["score_ulm"], adata.obsm["score_ora"]
```

## `dc.mt.consensus` — combine method scores

```python
dc.mt.consensus(result, verbose=False)
```

Computes a single consensus activity by aggregating the per-method scores (a signed
z-score mean across methods), with its own p-values.

- Pass an **AnnData** that already has `score_*` in `.obsm` → writes
  `score_consensus` / `padj_consensus`, returns `None`.
- Pass the **dict** returned by `decouple` (DataFrame input) → returns
  `(es, pv)`.
- Shortcut: `dc.mt.decouple(..., cons=True)` runs the methods and adds the consensus
  in one call.

```python
# One-shot: run several methods and get the consensus
dc.mt.decouple(adata, net, methods=["ulm", "mlm", "ora"], cons=True, tmin=3)
adata.obsm["score_consensus"]
```

## When to use this

- **Robustness / reduce method bias** — consensus across methods is a common default
  for TF activity when you don't want to defend a single choice.
- **Comparing methods** — `decouple` gives you every method's scores side by side in
  one object.

## Footguns

- Consensus needs **at least two** method scores present; running it after a single
  method is meaningless.
- Mixing methods with very different ranges is fine (consensus standardizes), but keep
  the **same net** across methods so sources line up.
- Per-method reproducibility still applies — set `seed` for `gsea`/`waggr` via
  `args={"gsea": {"seed": 0}}`.

## Related references

[methods.md](methods.md) (per-method kwargs, which to include),
[io-and-outputs.md](io-and-outputs.md) (dict vs `.obsm` returns),
[benchmarking.md](benchmarking.md) (deciding which methods win on your data).
