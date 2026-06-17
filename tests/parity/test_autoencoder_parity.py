"""Parity + leakage tests for :class:`PCAAutoencoderDetector` (autoencoder group).

Pins the PCA reconstruction-error autoencoder against an INDEPENDENT raw
scikit-learn computation and enforces the full-sample-leakage discipline:

- ``reconstruction_error`` must equal a raw :class:`sklearn.decomposition.PCA`
  ``inverse_transform`` reconstruction MSE (squared error) to ``1e-10``.
- the scaler, the PCA basis, AND the error-quantile threshold are fitted on the
  TRAIN slice ONLY; mutating OOS bars after time ``t`` never changes the score
  at ``t``; the threshold is a pure function of the train slice.

This is a NEW file authored by the autoencoder group (the shared
``test_sklearn_parity.py`` carries a skipped placeholder owned by the harness).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from anomaly_detector._exceptions import NotFittedError, ValidationError
from anomaly_detector.detectors.autoencoder import PCAAutoencoderDetector
from anomaly_detector.detectors.result import AnomalyResult

pytestmark = pytest.mark.parity

_FEATURES = ("ret", "vol", "zscore", "range", "autocorr")


def _make_features(returns: pd.Series, *, window: int = 21) -> pd.DataFrame:
    """Build a small causal-ish feature frame from a return series.

    Not the production feature engineer (that is a different group); a
    self-contained, NaN-free matrix is all the parity/leakage tests need.
    """
    ret = returns.astype("float64")
    roll = ret.rolling(window, min_periods=window)
    frame = pd.DataFrame(
        {
            "ret": ret,
            "vol": roll.std(),
            "zscore": (ret - roll.mean()) / roll.std(),
            "range": ret.abs().rolling(window, min_periods=window).max(),
            "autocorr": ret.rolling(window, min_periods=window).apply(
                lambda w: pd.Series(w).autocorr(lag=1), raw=False
            ),
        }
    ).dropna()
    return frame


def _split(frame: pd.DataFrame, frac: float = 0.6) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = int(len(frame) * frac)
    return frame.iloc[:cut], frame.iloc[cut:]


def test_reconstruction_matches_raw_sklearn_inverse_transform(
    injected_anomalies: object,
) -> None:
    """Reconstruction MSE must match raw PCA ``inverse_transform`` to 1e-10."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    returns = injected_anomalies.returns  # type: ignore[attr-defined]
    frame = _make_features(returns)
    train, test = _split(frame)

    n_components = 3
    detector = PCAAutoencoderDetector(n_components=n_components, contamination=0.05, seed=7)
    detector.fit(train)
    ours = detector.reconstruction_error(test)

    # Independent raw-sklearn oracle: scaler + PCA fitted on TRAIN, reconstruct OOS.
    scaler = StandardScaler().fit(train.to_numpy(dtype="float64"))
    scaled_train = scaler.transform(train.to_numpy(dtype="float64"))
    pca = PCA(n_components=n_components, svd_solver="full", random_state=7).fit(scaled_train)

    scaled_test = scaler.transform(test.to_numpy(dtype="float64"))
    recon = pca.inverse_transform(pca.transform(scaled_test))
    expected = np.sum(np.square(scaled_test - recon), axis=1)

    np.testing.assert_allclose(ours, expected, atol=1e-10, rtol=0.0)


def test_threshold_is_train_quantile_only(injected_anomalies: object) -> None:
    """The flag threshold equals the ``1 - contamination`` quantile of TRAIN errors."""
    returns = injected_anomalies.returns  # type: ignore[attr-defined]
    frame = _make_features(returns)
    train, test = _split(frame)

    contamination = 0.05
    detector = PCAAutoencoderDetector(n_components=3, contamination=contamination, seed=7)
    detector.fit(train)

    train_errors = detector.reconstruction_error(train)
    expected_threshold = float(np.quantile(train_errors, 1.0 - contamination))

    result = detector.score(test)
    assert isinstance(result, AnomalyResult)
    assert result.threshold == pytest.approx(expected_threshold, abs=1e-12)
    assert result.detector == "autoencoder"
    # Flags are exactly score > train threshold.
    np.testing.assert_array_equal(
        result.flags.to_numpy(), result.scores.to_numpy() > result.threshold
    )


def test_fit_on_train_only_no_oos_leakage(injected_anomalies: object) -> None:
    """Mutating OOS bars after time t never changes the score/threshold at t."""
    returns = injected_anomalies.returns  # type: ignore[attr-defined]
    frame = _make_features(returns)
    train, test = _split(frame)

    detector = PCAAutoencoderDetector(n_components=3, contamination=0.05, seed=7)
    detector.fit(train)
    base = detector.score(test)

    # Perturb the LAST OOS row only; the threshold (train-derived) and every
    # earlier score must be byte-identical (no full-sample refit).
    perturbed = test.copy()
    perturbed.iloc[-1] = perturbed.iloc[-1] + 10.0
    after = detector.score(perturbed)

    assert after.threshold == base.threshold
    np.testing.assert_allclose(
        after.scores.to_numpy()[:-1], base.scores.to_numpy()[:-1], atol=0.0, rtol=0.0
    )


def test_score_before_fit_raises() -> None:
    """Scoring before fitting is a programming error -> ``NotFittedError``."""
    detector = PCAAutoencoderDetector()
    frame = pd.DataFrame(np.ones((5, 3)), columns=["a", "b", "c"])
    with pytest.raises(NotFittedError):
        detector.score(frame)
    with pytest.raises(NotFittedError):
        detector.reconstruction_error(frame)


def test_n_components_exceeding_features_rejected() -> None:
    """``n_components`` larger than the feature count is rejected at fit time."""
    frame = pd.DataFrame(np.random.default_rng(0).standard_normal((40, 2)), columns=["a", "b"])
    detector = PCAAutoencoderDetector(n_components=3)
    with pytest.raises(ValidationError):
        detector.fit(frame)


def test_invalid_constructor_params_rejected() -> None:
    """Out-of-range constructor parameters raise :class:`ValidationError`."""
    with pytest.raises(ValidationError):
        PCAAutoencoderDetector(n_components=0)
    with pytest.raises(ValidationError):
        PCAAutoencoderDetector(contamination=0.0)
    with pytest.raises(ValidationError):
        PCAAutoencoderDetector(contamination=0.5)


def test_column_mismatch_rejected(injected_anomalies: object) -> None:
    """Scoring with mismatched feature columns is rejected."""
    returns = injected_anomalies.returns  # type: ignore[attr-defined]
    frame = _make_features(returns)
    train, test = _split(frame)
    detector = PCAAutoencoderDetector(n_components=3, contamination=0.05).fit(train)

    renamed = test.rename(columns={"ret": "RETURN"})
    with pytest.raises(ValidationError):
        detector.score(renamed)


def test_to_dict_is_json_describable(injected_anomalies: object) -> None:
    """``to_dict`` reports config and fitted state without raising."""
    returns = injected_anomalies.returns  # type: ignore[attr-defined]
    frame = _make_features(returns)
    train, _ = _split(frame)

    detector = PCAAutoencoderDetector(n_components=3, contamination=0.05, seed=11)
    pre = detector.to_dict()
    assert pre["fitted"] is False
    assert pre["detector"] == "autoencoder"

    detector.fit(train)
    post = detector.to_dict()
    assert post["fitted"] is True
    assert post["n_train"] == len(train)
    assert post["threshold"] is not None
    assert post["feature_names"] == list(_FEATURES)
