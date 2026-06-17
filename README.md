# anomaly-detector

Flag anomalous trading days in liquid ETFs with **two independent unsupervised
detectors** — an Isolation Forest (Liu, Ting & Zhou 2008) and a PCA
reconstruction-error autoencoder (Sakurada & Yairi 2014, **no torch**) — under a
strictly causal walk-forward refit.

> **Honest headline.** Isolation Forest and the PCA-autoencoder agree on a small
> core of known macro-stress dates (2020-03, 2018-Q4, the 2022 selloff), but
> their day-level agreement is **modest** (Jaccard ~0.3–0.5) and precision
> against a naive `|z-return| > 3` proxy label is **low**. Anomaly flags are
> **diagnostic, not tradable** — there is **no ground-truth label**, so **no
> alpha is claimed**.

## Why it does not leak

The single largest risk in unsupervised anomaly detection on time series is
**full-sample leakage**. Here the `StandardScaler`, the Isolation Forest, the
PCA basis, **and every threshold** (contamination quantile, autoencoder
error cutoff) are fitted on the **TRAIN slice only**, then transform/score the
**disjoint out-of-sample slice**. A day's flag uses only information available
**strictly before** that day (the `.shift(1)` chokepoint). Four Hypothesis
property tests pin this:

1. **future-perturbation invariance** — mutating bars after day `t` never
   changes the score/flag at `t`;
2. **prefix-determinism** — a prefix scores identically to the full series
   restricted to that prefix;
3. **scale-invariance** of the z-features;
4. **monotonicity** of the flag count in `contamination`.

## Install

```bash
uv venv
uv pip install -e '.[data,viz,dev]'
```

## Quickstart

```bash
# Deterministic, no-network demo on a synthetic anomaly-injected series.
uv run anomaly-detector demo

# Scan a real ticker (Polygon EOD, degrades to synthetic on any failure).
uv run anomaly-detector scan SPY --start 2015-01-01 --end 2023-12-31 --detector both
```

## How it works

1. **Features** (`features/engineer.py`) — a causal per-day feature vector
   (log-return, rolling realized vol, return z-score, range/ATR, volume z-score,
   short return-autocorrelation), all from `.shift(1)`-safe rolling windows.
2. **Detectors** (`detectors/`) — `IsolationForestDetector`
   (score = `-score_samples`) and `PCAAutoencoderDetector`
   (score = reconstruction MSE), each fitted on TRAIN and scoring OOS, returning
   the shared frozen `AnomalyResult`.
3. **Evaluation** (`evaluation/`) — Jaccard/overlap agreement, regime alignment
   with known stress windows, precision/recall against the transparent
   `|z-return| > 3` proxy, and an optional, clearly-labeled-diagnostic
   fade-the-anomaly overlay with a Deflated Sharpe over the full grid
   (`n_trials = #detectors × #contamination × #windows`).

## Validation

| Check | What it pins |
| --- | --- |
| parity | `IsolationForestDetector` vs raw sklearn `score_samples` to `1e-10`; PCA reconstruction vs sklearn `inverse_transform` to `1e-10` |
| property | the four leakage/correctness invariants above |
| regression | golden injected-anomaly recovery (no lookahead) + honest-headline guard (modest Jaccard, low proxy precision) |
| integration | full causal walk-forward run on the synthetic fixture |

Coverage gate: **≥ 85%**; `ruff` + strict `mypy` clean.

## Limitations

- **No ground-truth label.** "Anomaly" here means "reconstructs poorly / isolates
  easily relative to the calm TRAIN regime", not "labeled market event". Precision
  against any proxy is therefore low by construction, and the tool makes no alpha
  claim.
- **Fixed, survivorship-aware ETF set.** The deployed tool fits at request time
  on a small set of long-lived, highly-liquid ETFs (e.g. SPY); these instruments
  have negligible survivorship bias over the sample, which is why a fixed set is
  acceptable here.
- **Modest agreement is expected.** Two genuinely independent detectors *should*
  disagree on the margin; high agreement would be a red flag for a shared
  artifact, not a strength.

## References

- F. T. Liu, K. M. Ting, Z.-H. Zhou. *Isolation Forest.* ICDM 2008.
- M. Sakurada, T. Yairi. *Anomaly Detection Using Autoencoders with Nonlinear
  Dimensionality Reduction.* MLSDA 2014.
- D. H. Bailey, M. López de Prado. *The Deflated Sharpe Ratio.* (2014) — for the
  optional overlay's multiplicity-aware yardstick.

## License

MIT — see [LICENSE](LICENSE).
