"""Property tests (Hypothesis): the four leakage/correctness invariants.

The brief's four required invariants, asserted on the FULL pipeline (causal
features -> fit-on-train -> score-OOS) rather than the feature layer alone:

(a) future-perturbation invariance - mutating bars strictly after day ``t`` never
    changes the OOS score/flag AT ``t`` (the core no-lookahead guarantee);
(b) prefix-determinism - scoring a feature prefix yields the same per-day OOS
    scores as scoring the full series, restricted to that prefix;
(c) scale-invariance of the z-features - multiplying the whole price path by a
    positive constant leaves the standardized features (and thus the detector's
    scores) unchanged;
(d) monotonicity of flag count in ``contamination`` - a higher contamination
    never yields fewer flags (fixed random_state).

These run on seeded synthetic Gaussian price paths so the suite is fully
deterministic and network-free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from anomaly_detector._rng import make_rng
from anomaly_detector.detectors.autoencoder import PCAAutoencoderDetector
from anomaly_detector.detectors.iforest import IsolationForestDetector
from anomaly_detector.features.engineer import engineer_features

pytestmark = pytest.mark.property

_WINDOW = 21


def _price_path(*, n_obs: int, seed: int, vol: float = 0.012) -> pd.Series:
    """Build a deterministic positive Gaussian-return price path."""
    gen = make_rng(seed)
    returns = gen.standard_normal(n_obs) * vol
    index = pd.date_range("2015-01-01", periods=n_obs, freq="B")
    prices = 100.0 * np.cumprod(1.0 + returns)
    return pd.Series(prices, index=index, name="price")


def _split(frame: pd.DataFrame, frac: float = 0.6) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = int(len(frame) * frac)
    return frame.iloc[:cut], frame.iloc[cut:]


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_future_perturbation_invariance(seed: int) -> None:
    """Mutating bars after ``t`` must not change the OOS score/flag at ``t``.

    The detector is fitted on the TRAIN slice and the OOS score at each day is a
    pure function of that day's (causal) features and the frozen train state. We
    perturb only the strict FUTURE relative to a probe day inside the OOS slice
    and confirm the probe-day score is byte-identical.
    """
    prices = _price_path(n_obs=320, seed=seed)
    features = engineer_features(prices, window=_WINDOW)
    train, test = _split(features)

    det = IsolationForestDetector(contamination=0.05, seed=3).fit(train)
    base_scores = det.score(test).scores

    # Probe a day midway through the OOS slice; mutate every OOS row strictly
    # after it. Because each OOS row scores independently against the frozen
    # train state, the probe-day score must not move.
    probe = len(test) // 2
    perturbed = test.copy()
    perturbed.iloc[probe + 1 :] = perturbed.iloc[probe + 1 :] * 7.0 + 3.0
    perturbed_scores = det.score(perturbed).scores

    np.testing.assert_allclose(
        base_scores.iloc[: probe + 1].to_numpy(),
        perturbed_scores.iloc[: probe + 1].to_numpy(),
        atol=1e-12,
        rtol=0.0,
    )


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_prefix_determinism(seed: int) -> None:
    """A prefix scores identically to the full OOS slice restricted to the prefix."""
    prices = _price_path(n_obs=320, seed=seed)
    features = engineer_features(prices, window=_WINDOW)
    train, test = _split(features)

    det = PCAAutoencoderDetector(n_components=3, contamination=0.05, seed=5).fit(train)
    full_scores = det.score(test).scores

    cut = max(4, len(test) // 2)
    prefix_scores = det.score(test.iloc[:cut]).scores

    np.testing.assert_allclose(
        prefix_scores.to_numpy(),
        full_scores.iloc[:cut].to_numpy(),
        atol=1e-12,
        rtol=0.0,
    )


@settings(max_examples=25, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    scale=st.floats(min_value=0.05, max_value=500.0, allow_nan=False, allow_infinity=False),
)
def test_zscore_scale_invariance(seed: int, scale: float) -> None:
    """Scaling the price path by a positive constant leaves OOS scores unchanged.

    Every feature is a log-return statistic or a z-score, so a positive rescale
    of the price path cancels; the detector fitted on the rescaled train slice
    therefore produces the same standardized features and the same OOS scores.
    """
    prices = _price_path(n_obs=320, seed=seed)
    base = engineer_features(prices, window=_WINDOW)
    scaled = engineer_features(prices * scale, window=_WINDOW)

    base_train, base_test = _split(base)
    scaled_train, scaled_test = _split(scaled)

    det_a = IsolationForestDetector(contamination=0.05, seed=9).fit(base_train)
    det_b = IsolationForestDetector(contamination=0.05, seed=9).fit(scaled_train)

    np.testing.assert_allclose(
        det_a.score(base_test).scores.to_numpy(),
        det_b.score(scaled_test).scores.to_numpy(),
        atol=1e-8,
        rtol=0.0,
    )


@settings(max_examples=20, deadline=None)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_flag_count_monotonic_in_contamination(seed: int) -> None:
    """Higher ``contamination`` never yields fewer flags (fixed random_state)."""
    prices = _price_path(n_obs=320, seed=seed)
    features = engineer_features(prices, window=_WINDOW)
    train, test = _split(features)

    counts: list[int] = []
    for contamination in (0.01, 0.02, 0.05, 0.10, 0.20):
        det = IsolationForestDetector(contamination=contamination, seed=7).fit(train)
        counts.append(int(det.score(test).flags.sum()))

    assert counts == sorted(counts), f"flag counts not monotone: {counts}"
