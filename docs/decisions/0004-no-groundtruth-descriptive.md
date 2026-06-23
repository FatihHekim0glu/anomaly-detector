# ADR-0004: No ground-truth label, so the headline is descriptive, not predictive

- **Status:** Accepted
- **Date:** 2026-06-17
- **Deciders:** anomaly-detector maintainers
- **Related:** [ADR-0005](0005-proxy-label.md) (the transparent proxy label and
  why precision against it is low)

## Context

There is **no ground-truth set of "true" anomalous trading days**. "Anomaly" is
not an observable event class; it is a model-relative judgement ("reconstructs
poorly / isolates easily relative to the calm TRAIN regime"). Any label we could
construct (`|z-return| > 3`, VIX spikes, a hand-picked list of crash dates) is
itself a heuristic, not a gold standard.

This creates a strong temptation to over-claim. With two detectors and a few
tunable knobs it is easy to report a precision/recall against *some* proxy, frame
it as detection skill, and slide from "these days look unusual" to "this flags a
tradable signal". On unsupervised time-series anomaly detection that slide is the
classic way honest research turns into a backtest-overfit illusion.

## Decision

**The headline is explicitly DESCRIPTIVE and the tool claims no alpha.** The
reported quantities are agreement and stability, not predictive skill:

- **Jaccard** between the two independent detectors' OOS flag sets: *how much do
  two genuinely different methods agree?*
- **Regime alignment**: *do the flags overlap known macro-stress windows?*
- Precision/recall against a **transparent, clearly-labelled proxy** (not a
  ground truth), reported alongside an explicit statement that low precision is
  the expected, honest outcome.

The verdict/summary text must **never** imply a profitable or tradable signal.
The flags are diagnostic. This is mechanically enforced: a regression guard runs
the shipped causal walk-forward path and pins its Jaccard inside `[0.68, 0.78]`
and caps proxy precision at `0.10`, so the summary cannot silently drift into
"the flags cleanly predict the proxy".

## Consequences

- **Positive.** The project is honest by construction; there is no path by which a
  parameter sweep turns the headline into a fake signal without a test going red.
- **Positive.** The descriptive framing is *more* defensible than a predictive one
  would be: modest agreement between independent detectors is a real, reportable
  finding that needs no ground truth.
- **Cost.** The tool cannot answer "is today tradeably anomalous?", only "do
  independent detectors agree today looks unusual, and does that line up with
  known stress?". This is the correct scope given the absence of labels.
- **Risk addressed.** "Manufacture a label, report precision, imply a signal",
  the standard over-claim on unlabelled anomaly detection, is rejected and
  guarded.
