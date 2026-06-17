"""Unit tests for the toy fade-the-anomaly overlay (DIAGNOSTIC ONLY).

These verify the overlay is strictly causal (no flag acted on before it is
known), that costs and trade counts are accounted honestly, that the Deflated
Sharpe falls as the multiplicity ``n_trials`` grows (the multiple-testing
penalty), and that the result serializes cleanly. The overlay makes NO alpha
claim; these tests pin its mechanics and the honest-yardstick behaviour, not any
profitability.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from anomaly_detector._exceptions import ValidationError
from anomaly_detector.evaluation.overlay import OverlayResult, fade_the_anomaly_overlay

pytestmark = pytest.mark.unit


def _series(values: list[float], start: str = "2020-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.Series(values, index=idx, name="return", dtype="float64")


def test_overlay_is_causal_position_shifted() -> None:
    """The position is earned on the NEXT day's return (flags.shift(1))."""
    # Day 1 has a big positive return and is flagged; the fade position is short
    # and is earned on day 2's return only.
    returns = _series([0.0, 0.05, -0.03, 0.0, 0.0])
    flags = pd.Series([False, True, False, False, False], index=returns.index)
    result = fade_the_anomaly_overlay(flags, returns, n_trials=1, cost_bps=0.0)

    # No flag is acted on its own day: day-1 (the flag day) earns nothing.
    assert result.oos_returns.iloc[1] == pytest.approx(0.0)
    # Day 2 earns the fade: position = -sign(r_day1) = -1, times r_day2 = -0.03,
    # so the overlay return is +0.03 (faded the up-move into a down-day).
    assert result.oos_returns.iloc[2] == pytest.approx(0.03)
    # All other (non-traded) days are exactly zero.
    assert result.oos_returns.iloc[0] == 0.0
    assert result.oos_returns.iloc[3] == 0.0


def test_overlay_future_perturbation_invariance() -> None:
    """Mutating bars AFTER a trade day never changes earlier overlay returns."""
    returns = _series([0.0, 0.04, -0.02, 0.01, 0.0, 0.0])
    flags = pd.Series([False, True, False, False, False, False], index=returns.index)
    base = fade_the_anomaly_overlay(flags, returns, n_trials=1, cost_bps=0.0)

    mutated = returns.copy()
    mutated.iloc[4] = 0.99  # change a bar strictly after the day-2 earned return
    mutated.iloc[5] = -0.99
    perturbed = fade_the_anomaly_overlay(flags, mutated, n_trials=1, cost_bps=0.0)

    # Overlay returns up to and including the earned day are unchanged.
    pd.testing.assert_series_equal(base.oos_returns.iloc[:4], perturbed.oos_returns.iloc[:4])


def test_overlay_costs_reduce_returns() -> None:
    """A positive per-side cost lowers the net return on traded days."""
    returns = _series([0.0, 0.05, -0.03, 0.0])
    flags = pd.Series([False, True, False, False], index=returns.index)
    free = fade_the_anomaly_overlay(flags, returns, n_trials=1, cost_bps=0.0)
    costed = fade_the_anomaly_overlay(flags, returns, n_trials=1, cost_bps=10.0)
    # With costs, the total net return is strictly lower (turnover is charged).
    assert costed.oos_returns.sum() < free.oos_returns.sum()


def test_overlay_no_flags_is_flat() -> None:
    """No flags -> no trades, an all-zero return series, NaN-or-finite Sharpe."""
    returns = _series([0.01, -0.02, 0.03, -0.01, 0.0])
    flags = pd.Series(False, index=returns.index)
    result = fade_the_anomaly_overlay(flags, returns, n_trials=1)
    assert result.n_trades == 0
    assert (result.oos_returns == 0.0).all()


def test_overlay_single_observation_has_nan_metrics() -> None:
    """A one-bar series cannot define a Sharpe; metrics fall back to NaN safely."""
    returns = _series([0.02])
    flags = pd.Series([True], index=returns.index)
    result = fade_the_anomaly_overlay(flags, returns, n_trials=1, cost_bps=0.0)
    # Too short for a Sharpe -> NaN descriptive and deflated Sharpe, no crash.
    assert np.isnan(result.sharpe)
    assert np.isnan(result.deflated_sharpe)
    assert result.oos_returns.shape[0] == 1


def test_overlay_n_trades_counts_position_days() -> None:
    """n_trades counts the (shifted) days the overlay actually holds a position."""
    returns = _series([0.0, 0.05, 0.0, -0.04, 0.0, 0.0])
    flags = pd.Series([False, True, False, True, False, False], index=returns.index)
    result = fade_the_anomaly_overlay(flags, returns, n_trials=1, cost_bps=0.0)
    # Two flag days -> two held (shifted) position days.
    assert result.n_trades == 2


# --------------------------------------------------------------------------- #
# Deflated-Sharpe n_trials guard                                              #
# --------------------------------------------------------------------------- #
def _flagged_overlay(n: int = 200, seed: int = 3) -> tuple[pd.Series, pd.Series]:
    """A longer synthetic flag/return pair so the DSR is well-defined."""
    rng = np.random.default_rng(seed)
    rets = rng.standard_normal(n) * 0.01
    idx = pd.date_range("2019-01-01", periods=n, freq="B")
    returns = pd.Series(rets, index=idx, name="return")
    # Flag ~5% of days at random (a realistic contamination tail).
    flag_pos = rng.choice(n, size=max(2, n // 20), replace=False)
    flags = pd.Series(False, index=idx, name="anomaly_flag")
    flags.iloc[flag_pos] = True
    return flags, returns


def test_deflated_sharpe_non_increasing_in_n_trials() -> None:
    """The DSR is non-increasing as the multiplicity n_trials grows."""
    flags, returns = _flagged_overlay()
    dsr_values = []
    for n_trials in (1, 4, 12, 36):
        result = fade_the_anomaly_overlay(flags, returns, n_trials=n_trials, cost_bps=0.0)
        assert result.n_trials == n_trials
        dsr_values.append(result.deflated_sharpe)
    # Each larger trial count deflates (or ties) the previous DSR.
    for earlier, later in itertools.pairwise(dsr_values):
        assert later <= earlier + 1e-12


def test_overlay_n_trials_threaded_to_result() -> None:
    """The full grid count (#detectors x #contamination x #windows) is recorded."""
    flags, returns = _flagged_overlay()
    # e.g. 2 detectors x 3 contamination levels x 4 windows = 24 trials.
    n_trials = 2 * 3 * 4
    result = fade_the_anomaly_overlay(flags, returns, n_trials=n_trials)
    assert result.n_trials == 24
    assert result.meta["diagnostic"] is True


def test_overlay_rejects_bad_n_trials() -> None:
    """n_trials < 1 is rejected (the DSR multiplicity must count at least one)."""
    flags, returns = _flagged_overlay()
    with pytest.raises(ValidationError):
        fade_the_anomaly_overlay(flags, returns, n_trials=0)


def test_overlay_rejects_negative_cost() -> None:
    """A negative per-side cost is rejected."""
    flags, returns = _flagged_overlay()
    with pytest.raises(ValidationError):
        fade_the_anomaly_overlay(flags, returns, n_trials=1, cost_bps=-1.0)


def test_overlay_to_dict_is_json_safe() -> None:
    """OverlayResult.to_dict serializes to JSON with finite-or-None scalars."""
    import json

    flags, returns = _flagged_overlay()
    result = fade_the_anomaly_overlay(flags, returns, n_trials=24, cost_bps=5.0)
    assert isinstance(result, OverlayResult)
    payload = result.to_dict()
    encoded = json.dumps(payload)  # must not raise
    assert isinstance(encoded, str)
    assert payload["n_trials"] == 24
    assert set(payload) == {
        "oos_returns",
        "sharpe",
        "deflated_sharpe",
        "n_trials",
        "n_trades",
        "meta",
    }
