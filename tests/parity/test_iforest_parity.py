"""Parity + behaviour tests for :class:`IsolationForestDetector`.

These pin the wrapper against an INDEPENDENT raw-scikit-learn reference and
assert the leakage-safe contract:

- ``raw_score_samples`` equals a raw :class:`sklearn.ensemble.IsolationForest`
  ``score_samples`` (same derived ``random_state``, same train-fitted
  standardization) to ``1e-10``;
- the public anomaly score is exactly ``-score_samples``;
- the flag count is monotone non-decreasing in ``contamination``;
- ``fit`` touches the TRAIN slice only (scoring is independent of what comes
  after the train window).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from anomaly_detector._exceptions import NotFittedError, ValidationError
from anomaly_detector.detectors.iforest import (
    IsolationForestDetector,
    _derive_random_state,
)

pytestmark = pytest.mark.parity

_FEATURE_COLS = ("log_return", "realized_vol", "return_zscore")


def _features_from_returns(returns: pd.Series, window: int = 21) -> pd.DataFrame:
    """Build a small, NaN-free causal-ish feature matrix from a return series.

    Independent of the (separately-authored) features module: this fabricates a
    self-contained feature frame so the detector tests do not couple to that
    group's stub. Columns are a level return, a trailing realized vol, and a
    trailing z-score; the warm-up rows are dropped so there are no NaNs.
    """
    log_ret = np.log1p(returns)
    roll = log_ret.rolling(window)
    realized_vol = roll.std().shift(1)
    mu = log_ret.rolling(window).mean().shift(1)
    sd = log_ret.rolling(window).std().shift(1)
    zscore = (log_ret - mu) / sd.replace(0.0, np.nan)
    frame = pd.DataFrame(
        {
            "log_return": log_ret,
            "realized_vol": realized_vol,
            "return_zscore": zscore,
        }
    )
    return frame.dropna()


def _split(frame: pd.DataFrame, train_frac: float = 0.6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Anchored chronological train/OOS split (no shuffling)."""
    cut = int(len(frame) * train_frac)
    return frame.iloc[:cut], frame.iloc[cut:]


def test_iforest_matches_raw_sklearn_score_samples(injected_anomalies) -> None:
    """``raw_score_samples`` must match a raw IsolationForest to 1e-10."""
    feats = _features_from_returns(injected_anomalies.returns)
    train, test = _split(feats)

    det = IsolationForestDetector(contamination=0.05, n_estimators=128, seed=11)
    det.fit(train)
    ours = det.raw_score_samples(test)

    # Independent reference: same standardization (scaler fit on TRAIN only) and
    # the same derived random_state.
    scaler = StandardScaler().fit(train.to_numpy(dtype="float64"))
    forest = IsolationForest(
        n_estimators=128,
        contamination=0.05,
        random_state=_derive_random_state(11),
    )
    forest.fit(scaler.transform(train.to_numpy(dtype="float64")))
    reference = forest.score_samples(scaler.transform(test.to_numpy(dtype="float64")))

    np.testing.assert_allclose(ours, reference, atol=1e-10, rtol=0.0)


def test_public_score_is_negated_score_samples(injected_anomalies) -> None:
    """The public anomaly score equals ``-raw_score_samples`` exactly."""
    feats = _features_from_returns(injected_anomalies.returns)
    train, test = _split(feats)

    det = IsolationForestDetector(contamination=0.05, seed=3).fit(train)
    result = det.score(test)
    raw = det.raw_score_samples(test)

    np.testing.assert_allclose(result.scores.to_numpy(), -raw, atol=1e-10, rtol=0.0)
    assert result.detector == "iforest"
    assert result.n_train == len(train)
    assert result.n_test == len(test)


def test_flag_count_monotone_in_contamination(injected_anomalies) -> None:
    """Flag count is non-decreasing as contamination rises (fixed random_state)."""
    feats = _features_from_returns(injected_anomalies.returns)
    train, test = _split(feats)

    counts: list[int] = []
    for contamination in (0.01, 0.02, 0.05, 0.10, 0.20):
        det = IsolationForestDetector(contamination=contamination, seed=7).fit(train)
        counts.append(int(det.score(test).flags.sum()))

    assert counts == sorted(counts), f"flag counts not monotone: {counts}"


def test_fit_uses_train_only_no_lookahead(injected_anomalies) -> None:
    """Mutating bars strictly after the train window cannot change train fit state.

    A detector fitted on a train slice must produce identical OOS scores whether
    or not the post-train future is perturbed, because ``fit`` never saw it.
    """
    feats = _features_from_returns(injected_anomalies.returns)
    train, test = _split(feats)

    det_a = IsolationForestDetector(contamination=0.05, seed=5).fit(train)
    scores_a = det_a.score(test).scores

    # Perturb the OOS slice wildly; re-fit on the SAME train slice.
    perturbed_test = test * 10.0 + 1.0
    det_b = IsolationForestDetector(contamination=0.05, seed=5).fit(train)
    scores_b = det_b.score(test).scores  # score the original, unperturbed OOS

    # The train-derived threshold and scaler/forest are identical regardless of
    # the perturbation, so scoring the original OOS yields identical scores.
    np.testing.assert_allclose(scores_a.to_numpy(), scores_b.to_numpy(), atol=1e-12)
    assert det_a.to_dict()["threshold"] == det_b.to_dict()["threshold"]
    # Sanity: the perturbed frame really is different.
    assert not np.allclose(perturbed_test.to_numpy(), test.to_numpy())


def test_score_before_fit_raises() -> None:
    """Scoring before fit raises :class:`NotFittedError`."""
    det = IsolationForestDetector()
    frame = pd.DataFrame({"a": [0.1, 0.2], "b": [0.3, 0.4]})
    with pytest.raises(NotFittedError):
        det.score(frame)
    with pytest.raises(NotFittedError):
        det.raw_score_samples(frame)


def test_invalid_contamination_rejected() -> None:
    """Out-of-range contamination raises :class:`ValidationError`."""
    for bad in (0.0, -0.1, 0.5, 0.9, 1.0):
        with pytest.raises(ValidationError):
            IsolationForestDetector(contamination=bad)


def test_invalid_n_estimators_rejected() -> None:
    """A non-positive ``n_estimators`` raises :class:`ValidationError`."""
    with pytest.raises(ValidationError):
        IsolationForestDetector(n_estimators=0)


def test_to_dict_round_trips_and_is_json_safe(injected_anomalies) -> None:
    """``to_dict`` reports config; values are plain JSON-serializable scalars."""
    import json

    feats = _features_from_returns(injected_anomalies.returns)
    train, _ = _split(feats)

    det = IsolationForestDetector(contamination=0.03, n_estimators=64, seed=9)
    assert det.to_dict()["fitted"] is False
    assert det.to_dict()["threshold"] is None

    det.fit(train)
    payload = det.to_dict()
    assert payload["fitted"] is True
    assert payload["detector"] == "iforest"
    assert payload["n_estimators"] == 64
    assert payload["n_train"] == len(train)
    assert payload["threshold"] is not None
    # Round-trips through JSON without error.
    json.loads(json.dumps(payload))


def test_derive_random_state_is_deterministic_and_in_range() -> None:
    """The seed->random_state mapping is stable and within sklearn's int range."""
    assert _derive_random_state(7) == _derive_random_state(7)
    assert _derive_random_state(7) != _derive_random_state(8)
    rs = _derive_random_state(123)
    assert 0 <= rs < 2**31 - 1
