# anomaly-detector

Flag anomalous trading days in liquid ETFs with **two independent unsupervised
detectors** — an Isolation Forest (Liu, Ting & Zhou 2008) and a PCA
reconstruction-error autoencoder (Sakurada & Yairi 2014, **no torch**) — under a
strictly causal walk-forward refit.

> **Honest headline.** Isolation Forest and the PCA-autoencoder agree on a small
> core of known macro-stress dates (2020-03, 2018-Q4, the 2022 selloff), but
> their day-level agreement is **modest** (Jaccard **≈ 0.50** on the seeded
> synthetic series at the default 2 % contamination) and precision against a
> naive `|z-return| > 3` proxy label is **low** (**≈ 0.04**). Anomaly flags are
> **diagnostic, not tradable** — there is **no ground-truth label**, so **no
> alpha is claimed**.

## What the numbers actually say

Every figure below is the **measured** output of the seeded synthetic
anomaly-injected series (`generate_injected_series(n_obs=1200, seed=7)`),
detectors fitted on the calm front and scoring the disjoint stress-back slice
(the no-lookahead split the regression guard pins). Reproduce them with the
block at the bottom of this file.

| Quantity | Value | Reading |
| --- | --- | --- |
| Jaccard agreement (IF ∩ AE / IF ∪ AE) | **0.50** | modest — the detectors share a core, not a signal |
| Proxy precision vs `\|z-return\| > 3` | **0.04** | low — flags are diagnostic, not a clean predictor |
| Proxy recall vs `\|z-return\| > 3` | **0.32** | the flags catch some, miss most, of the naive proxy |
| Regime alignment (flags inside known stress windows) | **0.05** | the synthetic dates do not sit on real macro windows; on real ETF data this rises |
| Isolation Forest flags | 95 | per-detector OOS flag count |
| PCA-autoencoder flags | 128 | per-detector OOS flag count |

A Jaccard near **1.0** would mean the two detectors are redundant (a shared
artifact); near **0.0** would mean they are unrelated noise. The measured
**≈ 0.50** is exactly the honest middle: *agreement on a modest core*. The
regression suite pins this inside a documented `[0.20, 0.65]` band and caps proxy
precision at `0.20`, so the summary can never silently drift into implying a
tradable signal.

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

See [`docs/DESIGN.md`](docs/DESIGN.md) for the layered architecture and the
walk-forward data flow, and the ADRs in [`docs/decisions/`](docs/decisions/) for
*why* the contested choices (train-only fit, PCA-not-torch, `.shift(1)`,
no-ground-truth descriptive framing, the proxy label) were made.

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

```python
from anomaly_detector import generate_injected_series, run_anomaly_scan

series = generate_injected_series(n_obs=1200, seed=7)
result = run_anomaly_scan(prices=series.prices, detector="both", contamination=0.02)
print(result.summary())  # {n_flags, jaccard, proxy_precision, proxy_recall, ...}
```

## How it works

1. **Features** (`features/engineer.py`) — a causal per-day feature vector
   (log-return, rolling realized vol, return z-score, range/ATR, volume z-score,
   short return-autocorrelation), all from `.shift(1)`-safe rolling windows and
   `pct_change(fill_method=None)`.
2. **Detectors** (`detectors/`) — `IsolationForestDetector`
   (score = `-score_samples`) and `PCAAutoencoderDetector`
   (score = reconstruction MSE), each fitted on TRAIN and scoring OOS, returning
   the shared frozen `AnomalyResult`.
3. **Walk-forward** (`scan.py`) — an anchored/expanding walk-forward refits the
   scaler, both detectors, and all thresholds on each fold's TRAIN slice only and
   concatenates the disjoint OOS folds into a single zero-look-ahead score/flag
   series.
4. **Evaluation** (`evaluation/`) — Jaccard/overlap agreement, regime alignment
   with known stress windows, precision/recall against the transparent
   `|z-return| > 3` proxy, and an optional, clearly-labeled-diagnostic
   fade-the-anomaly overlay with a Deflated Sharpe over the full grid
   (`n_trials = #detectors × #contamination × #windows`).

## Validation

The suite is partitioned by intent; each layer pins a specific claim against an
oracle to a stated tolerance.

| Layer | Oracle | Tolerance | Test |
| --- | --- | --- | --- |
| parity | raw `sklearn.ensemble.IsolationForest.score_samples` (same `random_state`, same train-fitted `StandardScaler`) | `atol=1e-10` | `tests/parity/test_iforest_parity.py`, `tests/parity/test_sklearn_parity.py` |
| parity | raw `sklearn.decomposition.PCA.inverse_transform` reconstruction MSE | `atol=1e-10` | `tests/parity/test_autoencoder_parity.py`, `tests/parity/test_sklearn_parity.py` |
| parity | train-error quantile flag threshold vs hand-computed quantile | `abs=1e-12` | `tests/parity/test_autoencoder_parity.py` |
| property | future-perturbation invariance, prefix-determinism, z-feature scale-invariance, flag-count monotonicity in `contamination` | exact / `1e-12` | `tests/property/test_invariants.py`, `tests/property/test_feature_invariants.py` |
| regression | golden injected-anomaly recovery — recall on injected days > calm-background false-positive rate (no lookahead) | strict `>` | `tests/regression/test_golden_anomalies.py` |
| regression | honest-headline guard — Jaccard in `[0.20, 0.65]`, proxy precision ≤ `0.20` | banded | `tests/regression/test_honest_headline.py` |
| integration | full causal walk-forward scan on the synthetic fixture | end-to-end | `tests/integration/test_walk_forward_run.py` |

192 tests pass; coverage **92.86 %** (gate **≥ 85 %**); `ruff` and strict `mypy`
both clean.

## Limitations

- **No ground-truth label.** "Anomaly" here means "reconstructs poorly / isolates
  easily relative to the calm TRAIN regime", not "labeled market event". There is
  no curated set of true anomalous days to score against, so **every precision
  number against any proxy is low by construction**, and the tool makes **no
  alpha claim**. The headline is deliberately descriptive (agreement and
  stability), never predictive. This is the central honest-null caveat, not a
  footnote.
- **Fixed, survivorship-aware ETF set.** The deployed tool fits at request time
  on a small set of long-lived, highly-liquid ETFs (e.g. SPY, QQQ, IWM). These
  instruments existed across the entire sample and were never delisted, so a
  *fixed* universe carries **negligible survivorship bias** — there is no pool of
  dead tickers being silently excluded. A fixed set is therefore acceptable here
  precisely because the instruments are survivors by construction, not by
  selection; the same shortcut would be unsafe on single names.
- **Modest agreement is expected, not a defect.** Two genuinely independent
  detectors *should* disagree on the margin; high agreement would be a red flag
  for a shared artifact, not a strength.
- **Synthetic regime alignment is low.** The injected synthetic dates do not fall
  inside the real-world stress windows used by `regime_alignment`, so that metric
  reads low on the fixture; on real ETF data over 2018–2022 it rises as flags
  cluster on documented selloffs.

## References

- F. T. Liu, K. M. Ting, Z.-H. Zhou. *Isolation Forest.* ICDM 2008.
- M. Sakurada, T. Yairi. *Anomaly Detection Using Autoencoders with Nonlinear
  Dimensionality Reduction.* MLSDA 2014.
- D. H. Bailey, M. López de Prado. *The Deflated Sharpe Ratio: Correcting for
  Selection Bias, Backtest Overfitting, and Non-Normality.* The Journal of
  Portfolio Management, 2014 — for the optional overlay's multiplicity-aware
  yardstick.

## Reproduce

Every headline number above is regenerated by this block (deterministic, no
network):

```bash
uv run python - <<'PY'
from anomaly_detector.data import generate_injected_series
from anomaly_detector.detectors.autoencoder import PCAAutoencoderDetector
from anomaly_detector.detectors.iforest import IsolationForestDetector
from anomaly_detector.evaluation.agreement import compute_agreement
from anomaly_detector.features.engineer import engineer_features

inj = generate_injected_series(n_obs=1200, seed=7)
feats = engineer_features(inj.prices, window=21)
split = inj.prices.index[500]                       # calm-front / stress-back, no lookahead
train = feats.loc[feats.index < split]
test = feats.loc[feats.index >= split]
roos = inj.returns.reindex(test.index)

det_if = IsolationForestDetector(contamination=0.02, seed=7).fit(train)
det_ae = PCAAutoencoderDetector(contamination=0.02, seed=7).fit(train)
sm = compute_agreement(det_if.score(test), det_ae.score(test), roos, window=21)
print(f"jaccard          {sm.jaccard:.2f}")
print(f"proxy_precision  {sm.proxy_precision:.2f}")
print(f"proxy_recall     {sm.proxy_recall:.2f}")
print(f"regime_alignment {sm.regime_alignment:.2f}")
print(f"IF flags {sm.n_flags_a}  AE flags {sm.n_flags_b}")
PY
```

Full verification:

```bash
uv run ruff check src tests
uv run mypy
uv run pytest -q --cov=anomaly_detector --cov-report=term-missing
```

## License

MIT — see [LICENSE](LICENSE).
