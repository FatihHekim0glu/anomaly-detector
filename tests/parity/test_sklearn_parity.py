"""Parity tests: detectors vs. raw scikit-learn references (to 1e-10).

These pin each detector against an INDEPENDENT raw-sklearn computation, driven by
the PRODUCTION causal feature engineer (so the parity holds on the exact feature
matrix the tool actually scores):

- ``IsolationForestDetector.raw_score_samples`` must equal a raw
  :class:`sklearn.ensemble.IsolationForest` ``score_samples`` (same derived
  ``random_state``, same train-fitted standardization) to ``1e-10``.
- ``PCAAutoencoderDetector.reconstruction_error`` must equal a raw
  :class:`sklearn.decomposition.PCA` ``inverse_transform`` reconstruction MSE to
  ``1e-10``.
"""

from __future__ import annotations

import numpy as np
import pytest

from anomaly_detector.detectors.autoencoder import PCAAutoencoderDetector
from anomaly_detector.detectors.iforest import (
    IsolationForestDetector,
    _derive_random_state,
)
from anomaly_detector.features.engineer import engineer_features

pytestmark = pytest.mark.parity

_WINDOW = 21


def _split(frame, frac: float = 0.6):
    """Anchored chronological train/OOS split (no shuffling)."""
    cut = int(len(frame) * frac)
    return frame.iloc[:cut], frame.iloc[cut:]


def test_iforest_matches_raw_sklearn_score_samples(injected_anomalies) -> None:
    """``raw_score_samples`` must match a raw IsolationForest to 1e-10."""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    features = engineer_features(injected_anomalies.prices, window=_WINDOW)
    train, test = _split(features)

    seed = 11
    det = IsolationForestDetector(contamination=0.05, n_estimators=128, seed=seed)
    det.fit(train)
    ours = det.raw_score_samples(test)

    scaler = StandardScaler().fit(train.to_numpy(dtype="float64"))
    forest = IsolationForest(
        n_estimators=128,
        contamination=0.05,
        random_state=_derive_random_state(seed),
    )
    forest.fit(scaler.transform(train.to_numpy(dtype="float64")))
    reference = forest.score_samples(scaler.transform(test.to_numpy(dtype="float64")))

    np.testing.assert_allclose(ours, reference, atol=1e-10, rtol=0.0)


def test_pca_reconstruction_matches_raw_sklearn_inverse_transform(
    injected_anomalies,
) -> None:
    """Reconstruction MSE must match raw PCA inverse_transform to 1e-10."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    features = engineer_features(injected_anomalies.prices, window=_WINDOW)
    train, test = _split(features)

    n_components = 3
    det = PCAAutoencoderDetector(n_components=n_components, contamination=0.05, seed=7)
    det.fit(train)
    ours = det.reconstruction_error(test)

    scaler = StandardScaler().fit(train.to_numpy(dtype="float64"))
    scaled_train = scaler.transform(train.to_numpy(dtype="float64"))
    pca = PCA(n_components=n_components, svd_solver="full", random_state=7).fit(scaled_train)

    scaled_test = scaler.transform(test.to_numpy(dtype="float64"))
    recon = pca.inverse_transform(pca.transform(scaled_test))
    expected = np.sum(np.square(scaled_test - recon), axis=1)

    np.testing.assert_allclose(ours, expected, atol=1e-10, rtol=0.0)
