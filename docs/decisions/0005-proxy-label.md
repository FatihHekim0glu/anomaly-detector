# ADR-0005: A transparent `|z-return| > 3` proxy label, reported as low-precision

- **Status:** Accepted
- **Date:** 2026-06-17
- **Deciders:** anomaly-detector maintainers
- **Related:** [ADR-0004](0004-no-groundtruth-descriptive.md) (no ground truth →
  descriptive headline), [ADR-0003](0003-shift1-walkforward.md) (the proxy is
  built from a causal z-score)

## Context

Even without a ground-truth label ([ADR-0004](0004-no-groundtruth-descriptive.md)),
readers reasonably ask "do the flags line up with *anything*?". We need a yardstick
that is (a) cheap, (b) fully transparent, and (c) honestly *not* a ground truth —
so that its numbers cannot be mistaken for detection skill. The hazard is picking
an opaque or tuned label that quietly flatters the detectors and lets the proxy
masquerade as truth.

## Decision

Use **`|causal z-return| > 3`** as the proxy positive class:

- A day is a "proxy anomaly" when its return, standardized by a **strictly
  trailing** rolling mean/std (the same `.shift(1)` causal z-score the features
  use, so it never peeks at the day it normalizes), is more than 3σ from zero.
- The threshold (`PROXY_Z_THRESHOLD = 3.0`) and the construction are **documented
  constants in code**, not tuned to the detectors.
- Precision/recall of the **union** of the two detectors' flags against this proxy
  are reported in the summary, with the explicit framing that **low precision is
  the honest expected outcome** — the proxy is a crude, naive heuristic, not a
  label the detectors should reproduce.

On the seeded synthetic series this yields proxy precision ≈ `0.04` and recall ≈
`0.32`: the detectors and the naive proxy overlap on a *core* but are far from
identical, which is exactly the point — if precision were high, the detectors
would just be re-deriving a one-line z-score rule and adding nothing.

## Consequences

- **Positive.** Fully transparent and causal: anyone can recompute the proxy from
  the documented constant; it cannot leak (trailing window) and cannot be accused
  of being secretly tuned.
- **Positive.** Low precision *supports* the honest headline rather than
  undermining it — it demonstrates the detectors are not trivially equivalent to a
  naive rule, and that the flags are diagnostic, not a clean predictor.
- **Cost.** The proxy is a weak yardstick; it cannot validate detection skill (by
  design — there is none to validate without ground truth). It is a sanity check,
  not an evaluation metric.
- **Risk addressed.** "Pick a flattering or opaque label and report a high
  precision as if it were skill" is rejected; the proxy is deliberately crude,
  causal, transparent, and framed as low-precision.
