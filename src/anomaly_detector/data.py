"""Data layer: synthetic anomaly-injected generator + EOD price loader.

Two responsibilities, both import-pure (heavy deps are imported lazily inside
functions):

1. :func:`generate_injected_series` — a deterministic, seeded base low-vol
   return series with VOLATILITY BURSTS and JUMPS injected at KNOWN indices, so
   the detectors have a recoverable core to be tested against (with no network).
   The known injection indices are returned alongside the series so the
   regression suite can assert recovery without any ground-truth leakage into
   the detectors themselves.

2. :func:`load_prices` — fetch real daily EOD closes for a single ticker via the
   existing Polygon provider (reused from the HRP infra), degrading to the
   deterministic synthetic path on any upstream failure, and reporting which
   source was used (``"polygon"`` | ``"synthetic"``).

NO-LOOKAHEAD: returns are differenced with ``pct_change(fill_method=None)`` via
:func:`anomaly_detector.data.compute_returns` (re-exported from the shared
return helper); prices are never forward-filled before differencing.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import pandas as pd

    from anomaly_detector._typing import PricesLike

#: Where a price/return series ultimately came from (for the API ``data_source``).
DataSource = Literal["polygon", "synthetic"]

#: Default fixed liquid-ETF universe the deployed tool fits on (single-ticker per
#: request; the set documents the survivorship justification in the README).
DEFAULT_TICKER: str = "SPY"


@dataclass(frozen=True, slots=True)
class InjectedSeries:
    """A synthetic return series with anomalies injected at KNOWN indices.

    Attributes
    ----------
    returns:
        The synthetic per-day return series (calm base + injected stress),
        indexed by date.
    prices:
        The corresponding price level series (cumulated from ``returns``),
        indexed by the same dates.
    vol_burst_idx:
        The positional indices (into ``returns``) where volatility bursts were
        injected.
    jump_idx:
        The positional indices where discrete jumps were injected.
    meta:
        Free-form JSON-serializable provenance (seed, base vol, burst/jump sizes).
    """

    returns: pd.Series
    prices: pd.Series
    vol_burst_idx: tuple[int, ...]
    jump_idx: tuple[int, ...]
    meta: dict[str, Any] = field(default_factory=dict)

    def known_anomaly_idx(self) -> tuple[int, ...]:
        """Return the sorted union of injected vol-burst and jump indices.

        Returns
        -------
        tuple[int, ...]
            The known anomalous positional indices, ascending and de-duplicated.

        Raises
        ------
        NotImplementedError
            Always — this is a typed stub for parallel authoring.
        """
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this series.

        Raises
        ------
        NotImplementedError
            Always — this is a typed stub for parallel authoring.
        """
        raise NotImplementedError


def generate_injected_series(
    *,
    n_obs: int = 1000,
    seed: int = 7,
    base_vol: float = 0.008,
    n_vol_bursts: int = 6,
    n_jumps: int = 6,
    burst_vol_mult: float = 5.0,
    jump_size: float = 0.06,
    start: date = date(2015, 1, 1),
) -> InjectedSeries:
    """Generate a deterministic return series with anomalies at KNOWN indices.

    Builds a calm Gaussian base return series (``base_vol`` per-day std), then
    injects ``n_vol_bursts`` short volatility-burst windows and ``n_jumps``
    discrete jumps at indices drawn from a seeded
    :func:`anomaly_detector._rng.make_rng` generator. The injection indices are
    recorded on the returned :class:`InjectedSeries` so tests can assert detector
    recovery WITHOUT feeding any label into the detectors.

    Parameters
    ----------
    n_obs:
        Number of trading days to generate.
    seed:
        Master RNG seed (deterministic; same seed -> byte-identical series).
    base_vol:
        Per-day standard deviation of the calm base regime.
    n_vol_bursts:
        Number of volatility-burst windows to inject.
    n_jumps:
        Number of discrete jumps to inject.
    burst_vol_mult:
        Multiplier applied to ``base_vol`` inside a burst window.
    jump_size:
        Absolute return size of an injected jump (sign is randomized).
    start:
        First date of the generated business-day index.

    Returns
    -------
    InjectedSeries
        The synthetic series with its known injection indices.

    Raises
    ------
    ValidationError
        If ``n_obs`` is too small for the requested injections, or any size
        parameter is non-positive.
    NotImplementedError
        Always — this is a typed stub for parallel authoring.
    """
    raise NotImplementedError


def load_prices(
    ticker: str,
    start: date,
    end: date,
    *,
    source_pref: Literal["auto", "polygon", "synthetic"] = "auto",
    seed: int = 7,
) -> tuple[pd.Series, DataSource]:
    """Load a single-ticker daily close series, degrading to synthetic.

    With ``source_pref="polygon"`` (or ``"auto"``) the real Polygon EOD provider
    is tried first; on ANY failure — and always for ``"synthetic"`` — a
    deterministic synthetic price series (from :func:`generate_injected_series`)
    is returned. The second element reports which path was taken so the API can
    surface a ``data_source`` badge.

    LAZY IMPORT: the Polygon provider (and ``httpx``) are imported inside this
    function, never at module import time.

    Parameters
    ----------
    ticker:
        The asset symbol (e.g. ``"SPY"``).
    start, end:
        Inclusive date range.
    source_pref:
        ``"auto"``/``"polygon"`` try Polygon then fall back; ``"synthetic"``
        forces the offline path.
    seed:
        Seed for the synthetic fallback (deterministic).

    Returns
    -------
    tuple[pandas.Series, DataSource]
        The daily close series and the source it came from.

    Raises
    ------
    ValidationError
        If ``ticker`` is empty or ``end <= start``.
    NotImplementedError
        Always — this is a typed stub for parallel authoring.
    """
    raise NotImplementedError


def compute_returns(prices: PricesLike) -> pd.Series:
    """Convert a price series to simple returns with no lookahead.

    NO-LOOKAHEAD REQUIREMENT: differenced with ``pct_change(fill_method=None)``
    — prices are NEVER forward-filled before differencing (ffill-then-diff
    manufactures spurious zero returns across gaps and leaks information). The
    leading NaN row is dropped.

    Parameters
    ----------
    prices:
        A price level series (or single-column price panel) indexed by date.

    Returns
    -------
    pandas.Series
        Simple returns with the leading NaN row removed.

    Raises
    ------
    ValidationError
        If ``prices`` is malformed.
    NotImplementedError
        Always — this is a typed stub for parallel authoring.
    """
    raise NotImplementedError
