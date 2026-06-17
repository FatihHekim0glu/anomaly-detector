# ADR-0003: A `.shift(1)` flag chokepoint under an anchored walk-forward refit

- **Status:** Accepted
- **Date:** 2026-06-17
- **Deciders:** anomaly-detector maintainers
- **Related:** [ADR-0001](0001-fit-on-train-only.md) (train-only fit), the
  vendored `backtest/walk_forward.py` (purge/embargo machinery)

## Context

Train-only fitting ([ADR-0001](0001-fit-on-train-only.md)) closes the *estimator*
leakage surface, but two further look-ahead surfaces remain:

1. **Within a feature row.** A rolling statistic that standardizes day `t` and
   *includes* day `t` lets the feature peek at the very return it normalizes.
2. **Across the sample over time.** A model fitted once on an early slice and
   applied forever does not adapt; conversely, naively re-fitting on a window that
   overlaps the day being scored re-opens the leakage.

We need a single, mechanically-checkable rule that makes the flag at day `t` a
function of data strictly **before** `t`, end to end.

## Decision

Adopt a **`.shift(1)` chokepoint** with two halves, under an **anchored
(expanding) walk-forward** refit:

- **Upstream (features).** Rolling mean/std used to standardize day `t` exclude
  `t` via `.shift(1)`, and the whole assembled feature frame is shifted one row,
  so every feature at `t` reflects data through `t − 1` only. Returns use
  `pct_change(fill_method=None)` so gaps never fabricate zero returns.
- **Downstream (flags) + walk-forward.** The train slice is anchored at row 0 and
  **expands** fold by fold; each OOS fold is the disjoint block immediately after
  the (growing) train slice, with `train_end` exclusive of the OOS rows. Detectors
  refit per fold on TRAIN only and flag OOS days against the train-derived
  threshold. Concatenating the OOS folds yields one out-of-sample flag series with
  zero look-ahead. The vendored purge/embargo collapses to the one-day gap the
  shift already enforces on these non-overlapping daily observations.

## Consequences

- **Positive.** "No look-ahead" becomes an operational, testable property:
  future-perturbation invariance (mutating bars at/after `t` never changes row
  `t`) and prefix-determinism are pinned by Hypothesis tests.
- **Positive.** The expanding window lets the detectors adapt as history grows
  while never scoring a day they were fitted on.
- **Cost.** The first feature rows are an unscored warm-up (rolling window +
  one-row shift), and each fold refits from scratch. Both are cheap on daily ETF
  data.
- **Cost / dependency.** The collapse of purge/embargo to a one-day gap is valid
  **only** for non-overlapping daily observations; weekly returns or an
  overlapping multi-day label would re-introduce a real embargo. This dependency
  is stated so it is not forgotten if the horizon ever changes.
- **Risk addressed.** Both the within-row peek and the across-time refit leak are
  closed by one rule that the tests can check directly.
