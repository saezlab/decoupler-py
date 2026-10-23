# Benchmarking methods and nets

`dc.bm` evaluates how well a method + net recover a known ground truth — typically a
perturbation dataset where you know which program was up/down in each sample. Use it
to **choose a method** or **validate a net**, not for routine scoring.

## `dc.bm.benchmark`

```python
dc.bm.benchmark(
    adata,                 # AnnData with a perturbation ground truth in .obs
    net,                   # a net, or a dict of {name: net} to compare several
    metrics=None,          # e.g. ["auroc", "auprc"] (None = defaults)
    groupby=None,          # obs column(s) to stratify the evaluation
    runby="expr",          # evaluate per-observation ("expr") or per-source
    sfilt=False,           # filter sources to those with a ground-truth label
    thr=0.1, emin=5,
    kws_decouple=None,     # kwargs forwarded to dc.mt.decouple (methods, args, tmin, ...)
    verbose=False,
)
```

- Internally runs `dc.mt.decouple`, so pass method selection and per-method args via
  `kws_decouple` (e.g. `kws_decouple={"methods": ["ulm", "mlm"], "tmin": 3}`).
- Pass a **dict of nets** to benchmark several priors head to head.
- The ground truth (which source is perturbed, and its sign) must be encoded in
  `adata.obs` — see the decoupler benchmarking tutorial for the expected columns.

## Metrics and plots

- `dc.bm.metric` — the scoring functions: `auc`, `fscore`, `qrank`, `hmean`,
  `dict_metric`.
- `dc.bm.pl` — plots of benchmark results: `auc`, `bar`, `fscore`, `qrank`, `summary`.

```python
res = dc.bm.benchmark(adata, {"collectri": net_a, "dorothea": net_b},
                      kws_decouple={"methods": ["ulm", "mlm"]})
dc.bm.pl.summary(res)   # compare methods × nets
```

## When to use

- Deciding **which method** to trust for your data/organism.
- Comparing **candidate nets** (e.g. CollecTRI vs DoRothEA levels).
- Sanity-checking a **custom net** against a perturbation benchmark
  (`dc.ds.knocktf` and similar provide ground-truth datasets).

## Footguns

- Benchmarking is only meaningful with a **real ground truth** — a toy net won't tell
  you anything.
- It re-runs scoring internally; keep `kws_decouple` consistent with how you intend to
  score for real, or the comparison won't transfer.

## Related references

[multi-method.md](multi-method.md) (`decouple`, which `benchmark` wraps),
[methods.md](methods.md) (candidate methods),
[priors.md](priors.md) (candidate nets).
