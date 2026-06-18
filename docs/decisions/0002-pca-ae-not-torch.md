# ADR-0002: The autoencoder is PCA reconstruction error, not a neural net

- **Status:** Accepted
- **Date:** 2026-06-17
- **Deciders:** anomaly-detector maintainers
- **Related:** [ADR-0001](0001-fit-on-train-only.md) (train-only fit applies to
  the PCA basis and its error-quantile threshold)

## Context

The second detector is described as an "autoencoder" because it follows the
Sakurada & Yairi (2014) recipe: learn a low-dimensional reconstruction of *calm*
data, then treat **reconstruction error** as the anomaly score: a day that
reconstructs poorly lies off the learned manifold and is anomalous. The obvious
implementation is a small neural autoencoder in torch or tensorflow.

That choice would impose a heavyweight, GPU-flavoured dependency on a package
whose entire value proposition is being a **pure, auditable, vendorable compute
core**. It would bloat the deployed container, introduce non-determinism (CUDA
kernels, thread-order float drift), make parity-to-`1e-10` impractical, and add
hundreds of megabytes for a model that, on a handful of daily features, a linear
method captures just as well.

## Decision

**Implement the autoencoder as PCA reconstruction error** (`sklearn.decomposition.PCA`):

```
score(x) = || x - pca.inverse_transform(pca.transform(x)) ||^2
```

PCA *is* the optimal linear autoencoder (encoder = projection onto the top
components, decoder = `inverse_transform`); the reconstruction-error semantics are
identical to the neural recipe, the implementation is a few lines, and it is
exactly reproducible. The flag threshold is a quantile of the TRAIN reconstruction
errors (per [ADR-0001](0001-fit-on-train-only.md)). **No torch, no tensorflow**:
not in the package, not in the `[data]`/`[viz]`/`[dev]` extras, not in the
deployed backend container.

## Consequences

- **Positive.** Zero heavy dependencies; the AE rides on the scikit-learn the
  Isolation Forest already needs.
- **Positive.** Deterministic and parity-testable: the reconstruction MSE matches
  raw `PCA.inverse_transform` to `1e-10`, and the threshold matches a
  hand-computed train quantile to `1e-12`.
- **Positive.** Fast enough to fit at request time, so the backend needs no
  pre-trained weights artifact.
- **Cost.** The detector is **linear**: it cannot model nonlinear manifolds a
  deep autoencoder might. This is acceptable here: the whole point is to pair a
  *genuinely different* second detector with the tree-based Isolation Forest, and
  the modest, honest agreement between them is the headline, not raw detection
  power. If nonlinear reconstruction ever became necessary, a kernel PCA variant
  would be preferred over re-introducing torch.
- **Risk addressed.** "Reach for a neural autoencoder by reflex", dragging in a
  GPU stack and non-determinism for no real gain on this feature set, is
  rejected.
