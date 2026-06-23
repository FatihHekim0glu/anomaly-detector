# Design

This document explains how `anomaly-detector` is put together: the layering, the
data flow through a single causal walk-forward fold, the invariants the compute
core guarantees, and the testing strategy that keeps the honest headline honest.
For *why* individual contested choices were made, see the numbered ADRs in
[`docs/decisions/`](decisions/).

## Goals and non-goals

**Goals**

- A pure, typed (`mypy --strict`, `py.typed`), side-effect-free compute core that
  can be audited line by line and vendored into a backend without dragging UI,
  network, or heavyweight ML dependencies along.
- Two **genuinely independent** unsupervised detectors, an Isolation Forest and
  a PCA reconstruction-error autoencoder, each parity-tested to `1e-10` against
  an independent scikit-learn reference.
- A **strictly causal** pipeline: the scaler, both detectors, and every threshold
  are fitted on the TRAIN slice only and score a disjoint future slice, so no day
  is flagged using information from its own future.
- An honest, **descriptive** verdict (detector agreement and stability, not
  Sharpe) that is *mechanically* prevented from over-claiming a tradable signal.

**Non-goals**

- A tradable signal. There is no ground-truth anomaly label, so the tool makes no
  alpha claim; "anomaly" means "reconstructs poorly / isolates easily relative to
  the calm TRAIN regime", not "labeled market event".
- A neural autoencoder. The "AE" is PCA reconstruction error (see
  [ADR-0002](decisions/0002-pca-ae-not-torch.md)); there is **no torch /
  tensorflow** anywhere in the package or the deployed container.
- A live trading system or a generic outlier-detection toolkit. The detectors
  exist to serve the *descriptive agreement* question.

## Layered architecture

The package is strictly layered; each layer imports only from the ones below it.
`src/` has **zero import-time side effects**, guarded by a subprocess
import-purity test (and scikit-learn / plotly / typer are imported lazily inside
the methods that need them).

```
            cli.py (Typer)            plots.py (Plotly)
                 |                          |
   ┌─────────────┴──────────────────────────┴──────────────────────┐
   │                          scan.py                               │
   │   run_anomaly_scan · ScanResult · anchored walk-forward refit  │
   ├────────────────────────────────────────────────────────────────
   │                        evaluation/                             │
   │     agreement.py · dsr.py · overlay.py                         │
   │  (Jaccard, proxy P/R, regime alignment · Deflated Sharpe ·     │
   │   optional fade-the-anomaly overlay, labeled diagnostic)       │
   ├────────────────────────────────────────────────────────────────
   │                         detectors/                             │
   │   iforest.py · autoencoder.py · result.py (frozen AnomalyResult)│
   ├────────────────────────────────────────────────────────────────
   │                          features/                             │
   │              engineer.py (causal .shift(1) features)           │
   ├────────────────────────────────────────────────────────────────
   │                          backtest/                             │
   │            walk_forward.py · costs.py · stats.py               │
   ├────────────────────────────────────────────────────────────────
   │   data.py · data_providers/        foundation (no internal deps)│
   │   (synthetic injector, Polygon     _validation · _constants ·   │
   │    EOD loader, returns)             _typing · _exceptions ·     │
   │                                     _manifest · _rng            │
   └────────────────────────────────────────────────────────────────
```

The same compute functions back the local CLI and the hosted FastAPI tool
unchanged; the backend vendors `src/anomaly_detector/` byte-for-byte behind a
`sys.path` shim and fits the detectors at request time (a few-year daily ETF
series through IsolationForest + PCA is cheap, with no pre-trained artifact and no
latency problem).

## Data flow through one causal fold

A single anchored walk-forward fold is the unit of work; `run_anomaly_scan`
concatenates the disjoint OOS folds into one zero-look-ahead series.

1. **Coerce** a price (or return) series to a single dated price path
   (`scan._coerce_prices`). A return input is integrated to a base-100 path so
   the same causal feature engineer and price-with-markers figure work unchanged.
2. **Engineer causal features** (`features/engineer.engineer_features`). Each
   per-day vector is a function of data strictly **before** that day: rolling
   statistics that standardize day `t` exclude day `t` (a `.shift(1)` on the
   rolling mean/std), and the whole assembled frame is shifted one row. Returns
   use `pct_change(fill_method=None)` so gaps never manufacture spurious zeros.
3. **Carve the fold** (`scan._walk_forward_folds`). The train slice is anchored
   at row 0 and **expands** fold by fold; each OOS fold is the disjoint block
   immediately after the growing train slice. `train_end` is exclusive of the OOS
   block, so train and OOS never overlap (the no-lookahead boundary).
4. **Fit on TRAIN only, score OOS.** Both detectors fit a `StandardScaler` and
   their model (forest / PCA basis) **and** derive their flag threshold from the
   TRAIN scores alone, then `transform`/`score` the disjoint OOS rows
   ([ADR-0001](decisions/0001-fit-on-train-only.md)).
5. **Flag with a `.shift(1)` chokepoint.** A day is flagged when its OOS anomaly
   score exceeds the train-derived threshold; because features are already causal
   and the threshold is train-only, the flag at `t` cannot depend on bar `t`'s own
   future ([ADR-0003](decisions/0003-shift1-walkforward.md)).
6. **Summarize descriptively** (`evaluation/agreement.compute_agreement`):
   Jaccard between the two OOS flag sets, precision/recall against the transparent
   `|z-return| > 3` proxy ([ADR-0005](decisions/0005-proxy-label.md)), and regime
   overlap with known stress windows. The summary is JSON-safe (every scalar
   through `_safe_float`, NaN/Inf coerced to `None`).

## Invariants the core guarantees

These are enforced by Hypothesis property tests, not just asserted in prose:

- **Future-perturbation invariance.** Mutating any bar at or after day `t` cannot
  change the feature vector, score, or flag at `t`. This is the operational
  definition of "no look-ahead".
- **Prefix-determinism.** Scoring a prefix yields exactly the same values as
  scoring the full series restricted to that prefix. There is no hidden global
  state.
- **Scale-invariance of the z-features.** Rescaling the input price/return series
  leaves the standardized features unchanged.
- **Monotonicity in `contamination`.** Raising the contamination quantile never
  *decreases* the flag count.

## Testing strategy

The suite is partitioned by intent so each claim has a home:

- **parity**: the two detectors reproduce raw scikit-learn references to
  `1e-10` (`-score_samples` for the forest; `inverse_transform` reconstruction
  MSE for the PCA AE), and the AE flag threshold matches a hand-computed train
  quantile to `1e-12`.
- **property**: the four invariants above (Hypothesis).
- **regression**: golden injected-anomaly recovery (recall on injected days
  strictly exceeds the calm-background false-positive rate, no lookahead) and the
  honest-headline guard on the shipped walk-forward path (Jaccard in
  `[0.68, 0.78]`, proxy precision <= `0.10`).
- **integration**: a full causal walk-forward scan on the synthetic fixture.

Coverage gate **>= 90 %** on core logic (currently **about 97 %**, with the
network EOD provider omitted); `ruff` and strict `mypy` clean.

## Honest-null discipline

The verdict is descriptive by construction. There is no curated set of true
anomalous days, so the tool reports *agreement and stability* (Jaccard, regime
overlap) rather than predictive skill, and any precision against the transparent
proxy is low by design ([ADR-0004](decisions/0004-no-groundtruth-descriptive.md)).
The regression band on Jaccard and the precision ceiling make it *mechanically
impossible* for the summary to drift into implying a profitable signal without a
test going red.
