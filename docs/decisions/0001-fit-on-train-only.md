# ADR-0001: Scaler, detectors, and all thresholds fit on the TRAIN slice only

- **Status:** Accepted
- **Date:** 2026-06-17
- **Deciders:** anomaly-detector maintainers
- **Related:** [ADR-0003](0003-shift1-walkforward.md) (the `.shift(1)` flag
  chokepoint and walk-forward refit)

## Context

The single largest failure mode in unsupervised anomaly detection on time series
is **full-sample leakage**. A `StandardScaler`, an `IsolationForest`, a PCA
basis, or a flag threshold fitted on the *entire* sample has already "seen" the
anomalous days it is later asked to detect. The standardization absorbs the
stress-period variance, the forest's split structure adapts to the outliers, the
PCA subspace bends toward them, and the contamination/error-quantile threshold is
computed over a distribution that *includes* the days it is supposed to flag.

The result is an in-sample anomaly score that looks impressively sharp and is
entirely circular. Because there is no ground-truth label to catch it, this
leakage is **silent**: nothing fails, the flags just quietly become a description
of the data the model was fitted on rather than a forward judgement.

## Decision

**Every fitted object is fitted on the TRAIN slice only and then scores the
disjoint OOS slice.** Concretely, on each walk-forward fold:

- the `StandardScaler` is `fit` on TRAIN rows and `transform`s OOS rows;
- the `IsolationForest` is `fit` on standardized TRAIN rows; the OOS score is
  `-score_samples` of the already-fitted forest;
- the PCA basis is `fit` on standardized TRAIN rows; the OOS score is the
  reconstruction MSE against the frozen basis;
- **both flag thresholds** (the Isolation Forest's `1 - contamination` score
  quantile and the autoencoder's train-error quantile) are computed from TRAIN
  scores **only**, never from OOS or full-sample scores.

No object is ever `fit` on a slice it will later score.

## Consequences

- **Positive.** The anomaly score at day `t` is an honest out-of-sample
  judgement: it cannot have been inflated by the model having fitted on `t`.
- **Positive.** The discipline is testable. A future-perturbation-invariance
  property test mutates bars after `t` and asserts the score/flag at `t` is
  unchanged; this fails loudly if any fit leaks across the boundary.
- **Positive.** Parity tests reproduce the exact behaviour with raw scikit-learn
  fitted on the same TRAIN slice with the same `random_state`, to `1e-10`.
- **Cost.** The earliest rows are spent as an unscored TRAIN warm-up, and each
  walk-forward fold refits from scratch. On a few-year daily ETF series this is
  cheap (sub-second), so there is no latency or pre-trained-artifact requirement
  for the deployed backend.
- **Risk addressed.** "Fit on the full sample because it's one line shorter", the
  default that silently inflates every in-sample anomaly metric, is rejected and
  guarded against.
