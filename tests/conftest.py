"""Shared, seeded test fixtures.

Every fixture is deterministic (driven by :func:`anomaly_detector._rng.make_rng`)
and returns pandas objects, so tests across the suite share identical synthetic
data with known structure:

- ``clean_series`` — a calm, low-volatility return/price series with NO injected
  anomalies (the well-behaved baseline; detectors should flag very little).
- ``injected_anomalies`` — a base series with volatility bursts and discrete
  jumps injected at KNOWN positional indices (exposed on the fixture object), so
  regression tests can assert detector recovery without leaking any label into
  the detectors.
- ``pure_noise`` — an i.i.d. Gaussian return series with no temporal structure
  (the null: agreement between detectors should be near chance).

Importing this module has no side effects beyond fixture registration.
"""

from __future__ import annotations

from datetime import date
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest

from anomaly_detector._rng import make_rng

#: Master seed for every fixture (date-stamped for provenance).
_SEED = 20260617

#: Default series length and rolling window used across fixtures.
_N_OBS = 1000
_WINDOW = 21


class InjectedFixture(NamedTuple):
    """A synthetic series plus the KNOWN indices where anomalies were injected.

    Attributes
    ----------
    prices:
        The price level series (cumulated from ``returns``), indexed by date.
    returns:
        The per-day return series (calm base + injected stress), indexed by date.
    vol_burst_idx:
        Positional indices into ``returns`` where volatility bursts were injected.
    jump_idx:
        Positional indices where discrete jumps were injected.
    """

    prices: pd.Series
    returns: pd.Series
    vol_burst_idx: tuple[int, ...]
    jump_idx: tuple[int, ...]

    @property
    def known_idx(self) -> tuple[int, ...]:
        """Sorted, de-duplicated union of the injected indices."""
        return tuple(sorted(set(self.vol_burst_idx) | set(self.jump_idx)))


def _business_index(n_obs: int, start: date = date(2015, 1, 1)) -> pd.DatetimeIndex:
    """Return an ``n_obs``-long business-day index starting at ``start``."""
    return pd.date_range(start=start, periods=n_obs, freq="B")


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded PCG64 generator shared by tests that need raw randomness."""
    return make_rng(_SEED)


@pytest.fixture
def clean_series() -> pd.Series:
    """A calm, low-volatility return series with NO injected anomalies.

    Shape ``(1000,)``. Gaussian returns with a small constant per-day std, so a
    correctly-calibrated detector flags only a small contamination tail and the
    two detectors should agree on essentially nothing pathological.
    """
    gen = make_rng(_SEED)
    returns = gen.standard_normal(_N_OBS) * 0.008
    index = _business_index(_N_OBS)
    return pd.Series(returns, index=index, name="return")


@pytest.fixture
def injected_anomalies() -> InjectedFixture:
    """A base series with vol bursts and jumps injected at KNOWN indices.

    Shape ``(1000,)``. A calm Gaussian base (per-day std ``0.008``) into which
    six short volatility-burst windows and six discrete jumps are written at
    seeded, recorded positional indices. The indices live on the returned
    :class:`InjectedFixture` so regression tests can assert recovery WITHOUT
    feeding any label to the detectors.
    """
    gen = make_rng(_SEED + 1)
    base_vol = 0.008
    returns = gen.standard_normal(_N_OBS) * base_vol

    # Deterministic, well-separated injection indices in the back half of the
    # series (so a causal train/OOS split can place them out-of-sample).
    vol_burst_idx = tuple(int(i) for i in np.linspace(560, 940, 6).astype(int))
    jump_idx = tuple(int(i) for i in np.linspace(520, 900, 6).astype(int) + 7)

    # Volatility bursts: a 5-day window of inflated-variance returns.
    for start_i in vol_burst_idx:
        end_i = min(start_i + 5, _N_OBS)
        returns[start_i:end_i] = gen.standard_normal(end_i - start_i) * (base_vol * 5.0)

    # Discrete jumps: a single large signed return.
    for j in jump_idx:
        if 0 <= j < _N_OBS:
            sign = 1.0 if gen.random() < 0.5 else -1.0
            returns[j] = sign * 0.06

    index = _business_index(_N_OBS)
    ret_series = pd.Series(returns, index=index, name="return")
    prices = pd.Series(100.0 * np.cumprod(1.0 + returns), index=index, name="price")
    return InjectedFixture(
        prices=prices,
        returns=ret_series,
        vol_burst_idx=vol_burst_idx,
        jump_idx=jump_idx,
    )


@pytest.fixture
def pure_noise() -> pd.Series:
    """An i.i.d. Gaussian return series with no temporal structure (the null).

    Shape ``(1000,)``. Every observation is independent zero-mean Gaussian noise
    with constant variance, so there is no real anomaly to find and detector
    agreement should be no better than chance.
    """
    gen = make_rng(_SEED + 2)
    returns = gen.standard_normal(_N_OBS) * 0.01
    index = _business_index(_N_OBS)
    return pd.Series(returns, index=index, name="return")


@pytest.fixture
def default_window() -> int:
    """The default rolling feature window shared by feature/detector tests."""
    return _WINDOW
