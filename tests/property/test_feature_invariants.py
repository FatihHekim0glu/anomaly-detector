"""Property tests (Hypothesis) for the causal feature engineer.

Covers the feature-layer half of the brief's leakage/correctness invariants:

* **scale-invariance** — multiplying the whole price path by a positive constant
  leaves the standardized / log-return-derived features unchanged (the prices
  cancel in every ratio/log).
* **future-perturbation invariance** — mutating bars strictly after day ``t``
  never changes the feature row AT ``t`` (the upstream no-lookahead guarantee of
  the ``.shift(1)`` chokepoint).
* **no NaN leak past warm-up** — once the leading warm-up rows are dropped, the
  returned matrix is finite everywhere (no lookahead row is silently emitted as
  NaN, and no real row is silently dropped).
* **prefix-determinism (bonus)** — features on a prefix equal the full-series
  features restricted to that prefix; rolling+shift only ever look backward.

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
from anomaly_detector.features.engineer import (
    DEFAULT_WINDOW,
    FEATURE_NAMES,
    causal_return_zscore,
    engineer_features,
)

pytestmark = pytest.mark.property

#: Features whose construction is invariant to a positive rescaling of the price
#: path (every one is a log-return statistic or a z-score; the scale cancels).
_SCALE_INVARIANT = (
    "log_return",
    "realized_vol",
    "return_zscore",
    "range_atr",
    "return_autocorr",
)


def _price_path(*, n_obs: int, seed: int, vol: float = 0.012) -> pd.Series:
    """Build a deterministic positive Gaussian-return price path."""
    gen = make_rng(seed)
    returns = gen.standard_normal(n_obs) * vol
    index = pd.date_range("2015-01-01", periods=n_obs, freq="B")
    prices = 100.0 * np.cumprod(1.0 + returns)
    return pd.Series(prices, index=index, name="price")


@settings(max_examples=40, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    scale=st.floats(min_value=0.05, max_value=500.0, allow_nan=False, allow_infinity=False),
    window=st.integers(min_value=2, max_value=40),
)
def test_zscore_scale_invariance(seed: int, scale: float, window: int) -> None:
    """Scaling the price path by a positive constant leaves z-features unchanged."""
    prices = _price_path(n_obs=320, seed=seed)
    base = engineer_features(prices, window=window)
    scaled = engineer_features(prices * scale, window=window)

    assert base.index.equals(scaled.index)
    for col in _SCALE_INVARIANT:
        # Mathematically exact (prices cancel in every log/ratio); the only gap
        # is float64 round-off, amplified for tiny windows where the z-score
        # divides by a near-zero rolling std. Assert with a combined rel/abs
        # tolerance rather than a bare absolute one.
        assert np.allclose(
            base[col].to_numpy(),
            scaled[col].to_numpy(),
            rtol=1e-7,
            atol=1e-9,
            equal_nan=True,
        ), f"{col} not scale-invariant (col={col})"


@settings(max_examples=40, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    cut=st.floats(min_value=0.4, max_value=0.85, allow_nan=False),
    window=st.integers(min_value=2, max_value=30),
)
def test_future_perturbation_invariance(seed: int, cut: float, window: int) -> None:
    """Mutating bars after ``t`` must not change the feature row at ``t``."""
    n_obs = 320
    prices = _price_path(n_obs=n_obs, seed=seed)
    k = int(cut * n_obs)

    # Arbitrarily corrupt every bar strictly after position ``k``.
    perturbed = prices.copy()
    gen = make_rng(seed + 999)
    noise = gen.standard_normal(n_obs - (k + 1)) * 50.0
    perturbed.iloc[k + 1 :] = np.abs(perturbed.iloc[k + 1 :].to_numpy() + noise) + 1.0

    base = engineer_features(prices, window=window)
    pert = engineer_features(perturbed, window=window)

    cutoff = prices.index[k]
    common = base.index.intersection(pert.index)
    upto = common[common <= cutoff]
    if len(upto) == 0:
        return  # warm-up consumed everything at/below the cut; nothing to assert.

    max_abs_diff = float(np.nanmax(np.abs(base.loc[upto].to_numpy() - pert.loc[upto].to_numpy())))
    assert max_abs_diff == 0.0, f"future leak: row<=t changed by {max_abs_diff:.3e}"


@settings(max_examples=40, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    window=st.integers(min_value=2, max_value=40),
    use_volume=st.booleans(),
)
def test_no_nan_past_warmup(seed: int, window: int, use_volume: bool) -> None:
    """The returned matrix is finite everywhere, with the expected column order."""
    n_obs = 300
    prices = _price_path(n_obs=n_obs, seed=seed)
    volume: pd.Series | None = None
    if use_volume:
        gen = make_rng(seed + 7)
        volume = pd.Series(
            np.abs(gen.standard_normal(n_obs)) * 1e6 + 5e6,
            index=prices.index,
            name="volume",
        )

    features = engineer_features(prices, volume=volume, window=window)

    assert tuple(features.columns) == FEATURE_NAMES
    assert not bool(features.isna().to_numpy().any())
    assert bool(np.isfinite(features.to_numpy()).all())
    # The warm-up only drops a prefix: every surviving date is contiguous tail.
    assert features.index.is_monotonic_increasing
    assert len(features) < n_obs


@settings(max_examples=30, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    prefix_frac=st.floats(min_value=0.5, max_value=0.9, allow_nan=False),
    window=st.integers(min_value=2, max_value=30),
)
def test_prefix_determinism(seed: int, prefix_frac: float, window: int) -> None:
    """A prefix yields the same feature rows as the full series, on that prefix."""
    n_obs = 320
    prices = _price_path(n_obs=n_obs, seed=seed)
    p = int(prefix_frac * n_obs)

    full = engineer_features(prices, window=window)
    prefix = engineer_features(prices.iloc[:p], window=window)

    common = full.index.intersection(prefix.index)
    assert len(common) > 0
    max_abs_diff = float(
        np.nanmax(np.abs(full.loc[common].to_numpy() - prefix.loc[common].to_numpy()))
    )
    assert max_abs_diff == 0.0, f"prefix differs from full: {max_abs_diff:.3e}"


@settings(max_examples=40, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=10_000),
    window=st.integers(min_value=2, max_value=40),
)
def test_causal_zscore_excludes_current_day(seed: int, window: int) -> None:
    """``causal_return_zscore`` standardizes by stats ending at ``t-1`` only.

    Mutating returns strictly after ``t`` cannot change ``z_t`` (the mean/std at
    ``t`` come from the trailing window ending at ``t-1``; the numerator uses
    only ``r_t`` itself).
    """
    n_obs = 320
    gen = make_rng(seed)
    returns = pd.Series(
        gen.standard_normal(n_obs) * 0.01,
        index=pd.date_range("2015-01-01", periods=n_obs, freq="B"),
        name="return",
    )
    k = n_obs // 2

    perturbed = returns.copy()
    perturbed.iloc[k + 1 :] = perturbed.iloc[k + 1 :].to_numpy() + 5.0

    z_base = causal_return_zscore(returns, window=window)
    z_pert = causal_return_zscore(perturbed, window=window)

    max_abs_diff = float(
        np.nanmax(np.abs(z_base.iloc[: k + 1].to_numpy() - z_pert.iloc[: k + 1].to_numpy()))
    )
    assert max_abs_diff == 0.0, f"z-score future leak: {max_abs_diff:.3e}"


def test_window_validation() -> None:
    """``window < 2`` (and non-int) is rejected by both public entry points."""
    from anomaly_detector._exceptions import ValidationError

    prices = _price_path(n_obs=60, seed=1)
    returns = prices.pct_change(fill_method=None).dropna()

    with pytest.raises(ValidationError):
        engineer_features(prices, window=1)
    with pytest.raises(ValidationError):
        causal_return_zscore(returns, window=1)
    with pytest.raises(ValidationError):
        # bool is an int subclass but is not a valid window.
        engineer_features(prices, window=True)


def test_default_window_constant() -> None:
    """The exported default window matches the brief's 21-day rolling window."""
    assert DEFAULT_WINDOW == 21
    assert len(FEATURE_NAMES) == 6


def test_single_column_dataframe_accepted() -> None:
    """A single-column price DataFrame is squeezed and engineered like a Series."""
    prices = _price_path(n_obs=80, seed=3)
    frame = prices.to_frame(name="close")

    from_series = engineer_features(prices, window=10)
    from_frame = engineer_features(frame, window=10)

    assert from_frame.index.equals(from_series.index)
    assert np.allclose(
        from_frame.to_numpy(), from_series.to_numpy(), rtol=1e-12, atol=1e-12, equal_nan=True
    )


def test_multi_column_dataframe_rejected() -> None:
    """A price DataFrame with more than one column is a malformed input."""
    from anomaly_detector._exceptions import ValidationError

    prices = _price_path(n_obs=80, seed=4)
    two_col = pd.DataFrame({"a": prices, "b": prices * 2.0})

    with pytest.raises(ValidationError):
        engineer_features(two_col, window=10)


def test_fully_flat_series_yields_empty_frame() -> None:
    """A perfectly flat (zero-variance) path has no meaningful z-score.

    Constant prices produce all-zero log-returns; the causal z-score divides by
    a zero rolling std (masked to NaN), so every row is legitimately dropped by
    the warm-up ``dropna`` rather than emitting a spurious finite value.
    """
    index = pd.date_range("2015-01-01", periods=120, freq="B")
    flat = pd.Series(100.0, index=index, name="price")

    features = engineer_features(flat, window=10)

    assert features.empty
    assert tuple(features.columns) == FEATURE_NAMES


def test_flat_subwindow_autocorr_is_zero_not_nan() -> None:
    """A flat sub-stretch inside a moving path yields autocorr 0.0, not NaN.

    The series has global variance (so z-scores exist and rows survive), but a
    constant sub-window drives the rolling-correlation degenerate branch, which
    must collapse to ``0.0`` so the affected rows stay finite.
    """
    n_obs = 200
    gen = make_rng(11)
    returns = gen.standard_normal(n_obs) * 0.01
    # Freeze a 30-day stretch flat (zero return) to force a zero-variance window.
    returns[80:110] = 0.0
    index = pd.date_range("2015-01-01", periods=n_obs, freq="B")
    prices = pd.Series(100.0 * np.cumprod(1.0 + returns), index=index, name="price")

    features = engineer_features(prices, window=10)

    assert not features.empty
    assert not bool(features.isna().to_numpy().any())
    assert bool(np.isfinite(features.to_numpy()).all())
