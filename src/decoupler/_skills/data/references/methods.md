# Choosing and configuring a method

All methods share the call in [calling-convention.md](calling-convention.md). This
file covers **which method to pick** and each method's **extra kwargs**. List them
live with `dc.mt.show()`.

## The methods at a glance

| Method | Kind | Uses weights? | p-values? | Output range | Typical use |
|---|---|:--:|:--:|---|---|
| `ulm` | Univariate Linear Model | yes (signed) | yes | (-inf, inf) | **Default** for TF/pathway activity; fast, robust |
| `mlm` | Multivariate Linear Model | yes (signed) | yes | (-inf, inf) | Accounts for correlated regulons; needs full-rank net |
| `viper` | VIPER regulon analysis | yes (signed) | yes | (-inf, inf) | TF activity with pleiotropy/target-overlap correction |
| `waggr` | Weighted Aggregate | yes (signed) | yes | (-inf, inf) | wmean/wsum/median aggregate; permutation p-values |
| `zscore` | Z-score | yes (signed) | yes | (-inf, inf) | Simple standardized weighted mean |
| `gsea` | Gene Set Enrichment Analysis | no | yes | (-inf, inf) | Ranked enrichment of gene sets |
| `ora` | Over-Representation Analysis | no | yes | (-inf, inf) | Fisher test on top/bottom features |
| `aucell` | AUCell | no | no | (0, 1) | Rank-based gene-set activity per cell |
| `gsva` | Gene Set Variation Analysis | no | no | (-1, 1) | Sample-relative gene-set variation |
| `mdt` | Multivariate Decision Tree | yes | no | (0, 1) | Random-forest feature importance as activity |
| `udt` | Univariate Decision Tree | yes | no | (0, 1) | Single-tree importance as activity |

Rules of thumb:
- **Weighted net (has signed `weight`, e.g. CollecTRI/DoRothEA/PROGENy)** → prefer
  `ulm` (default), `mlm`, `viper`, or `waggr`.
- **Unweighted gene sets (e.g. MSigDB hallmarks, `.gmt`)** → `ora`, `gsea`, `aucell`,
  or `gsva`.
- **Unsure / want robustness** → run several and take a consensus
  ([multi-method.md](multi-method.md)).

## Method-specific kwargs

Pass these as extra keyword args to the call, e.g. `dc.mt.ora(adata, net, n_up=300)`.

- **`ulm`, `mlm`** — `tval=True`: return the t-value (default) vs the raw coefficient.
- **`ora`** — `n_up=None` (n top features as the "up" set; default derives from data),
  `n_bm=0` (n bottom features), `n_bg=20000` (background size for the Fisher test),
  `ha_corr=0.5` (Haldane–Anscombe continuity correction).
- **`gsea`** — `times=1000` (permutations), `seed=42`. **Reproducibility:** p-values
  come from permutations; fix `seed` (and keep `times` constant) for stable results.
- **`gsva`** — `kcdf="gaussian"`, `maxdiff=True`, `absrnk=False`, `tau=1`.
- **`aucell`** — `n_up=None`: size of the ranking cutoff (top-N genes per cell;
  default derives from the data).
- **`viper`** — `pleiotropy=True`, `reg_sign=0.05`, `n_targets=10`, `penalty=20`
  (pleiotropy correction for shared targets between regulons).
- **`zscore`** — `flavor="RoKAI"` (alternative standardization flavor).
- **`waggr`** — `fun="wmean"` (or `"wsum"`, `"median"`), `times=1000`, `seed=42`
  (permutation p-values; fix `seed` for reproducibility).
- **`mdt`, `udt`** — no extra kwargs.

## Notes / footguns

- **`ulm` is the recommended default** for TF and pathway activity in most workflows.
- **`mlm`** solves a single multivariate regression; if the net is rank-deficient
  (highly collinear regulons) it can be unstable — prefer `ulm`/`viper` then.
- **`mlm` p-values are not FDR-adjusted** the way other testing methods are (it is
  the one method the runner skips BH correction for); interpret `padj_mlm` accordingly.
- **Permutation methods (`gsea`, `waggr`)** are stochastic — set `seed` and hold
  `times` fixed, or scores/p-values will drift between runs.
- **Non-testing methods (`aucell`, `gsva`, `mdt`, `udt`)** produce no `padj_<method>`.

## Related references

[calling-convention.md](calling-convention.md) (shared args, input formats),
[multi-method.md](multi-method.md) (`decouple` / `consensus`),
[priors.md](priors.md) (weighted vs gene-set nets),
[io-and-outputs.md](io-and-outputs.md) (where scores land).
