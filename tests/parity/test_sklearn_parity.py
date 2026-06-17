"""Parity tests: detectors vs. raw scikit-learn references (to 1e-10).

These pin each detector against an INDEPENDENT raw-sklearn computation:

- ``IsolationForestDetector.raw_score_samples`` must equal a raw
  :class:`sklearn.ensemble.IsolationForest` ``score_samples`` (same seed,
  same standardized features) to ``1e-10``.
- ``PCAAutoencoderDetector.reconstruction_error`` must equal a raw
  :class:`sklearn.decomposition.PCA` ``inverse_transform`` reconstruction MSE to
  ``1e-10``.

Skipped until the detectors are implemented (the scaffold ships green). The
parallel author removes the skip and fills the body.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.parity


@pytest.mark.skip(reason="scaffold: implement once IsolationForestDetector.score is real")
def test_iforest_matches_raw_sklearn_score_samples() -> None:
    """``-raw_score_samples`` must match a raw IsolationForest to 1e-10."""
    raise NotImplementedError


@pytest.mark.skip(reason="scaffold: implement once PCAAutoencoderDetector.score is real")
def test_pca_reconstruction_matches_raw_sklearn_inverse_transform() -> None:
    """Reconstruction MSE must match raw PCA inverse_transform to 1e-10."""
    raise NotImplementedError
