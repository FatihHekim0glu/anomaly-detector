"""Causal per-day feature engineering for anomaly detection.

Turns a price (and optional volume) series into a per-day feature matrix that a
detector can score. Every feature is CAUSAL: the value at day ``t`` is built
only from information available strictly before ``t``, via ``.shift(1)``-safe
rolling windows. Returns are differenced with ``pct_change(fill_method=None)``
(never forward-filling prices, which would manufacture spurious zero returns and
leak information across gaps).

Feature set (per day):

- ``log_return``        — log of the gross return ``ln(p_t / p_{t-1})``.
- ``realized_vol``      — rolling realized volatility of log-returns.
- ``return_zscore``     — return standardized by its rolling mean/std.
- ``range_atr``         — rolling average true range proxy (or |return| ATR).
- ``volume_zscore``     — volume standardized by its rolling mean/std (0 if no
  volume supplied).
- ``return_autocorr``   — short-lag rolling autocorrelation of returns.

NO-LOOKAHEAD REQUIREMENT: the rolling statistics that *standardize* day ``t``
must exclude day ``t`` itself (a ``.shift(1)`` on the rolling mean/std), so the
z-scores cannot peek at the very return they normalize. This is the upstream
half of the ``.shift(1)`` flag chokepoint.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from anomaly_detector._typing import PricesLike

#: The ordered feature-column names produced by :func:`engineer_features`.
FEATURE_NAMES: tuple[str, ...] = (
    "log_return",
    "realized_vol",
    "return_zscore",
    "range_atr",
    "volume_zscore",
    "return_autocorr",
)

#: Default rolling window (in trading days) for the realized-vol / z-score /
#: autocorrelation statistics.
DEFAULT_WINDOW: int = 21


def engineer_features(
    prices: PricesLike,
    *,
    volume: pd.Series | None = None,
    window: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """Build the causal per-day feature matrix from a price (and volume) series.

    Each column is one of :data:`FEATURE_NAMES`, computed from
    ``.shift(1)``-safe rolling windows so that day ``t``'s features use only
    data strictly before ``t``. Rows that cannot be formed without lookahead
    (the leading ``window`` rows whose statistics would otherwise peek) are
    dropped, so the returned frame is shorter than ``prices`` by the warm-up.

    Parameters
    ----------
    prices:
        A price level series (or single-column price panel), indexed by date.
    volume:
        Optional volume series aligned to ``prices``; when ``None`` the
        ``volume_zscore`` column is filled with zeros.
    window:
        Rolling window length (trading days) for the realized-vol, z-score, and
        autocorrelation statistics. Must be ``>= 2``.

    Returns
    -------
    pandas.DataFrame
        A per-day feature matrix (rows = date, columns = :data:`FEATURE_NAMES`),
        free of NaN and with no lookahead.

    Raises
    ------
    ValidationError
        If ``prices`` is malformed or ``window < 2``.
    NotImplementedError
        Always — this is a typed stub for parallel authoring.
    """
    raise NotImplementedError


def causal_return_zscore(returns: pd.Series, *, window: int = DEFAULT_WINDOW) -> pd.Series:
    """Standardize ``returns`` by a strictly-trailing rolling mean and std.

    The mean and std used to standardize day ``t`` are computed over the window
    ENDING AT ``t - 1`` (a ``.shift(1)`` on the rolling statistics), so the
    z-score never sees the return it normalizes. Exposed separately because it
    is reused by the transparent proxy label (``|z-return| > 3``) in the
    evaluation layer.

    Parameters
    ----------
    returns:
        A per-day return series indexed by date.
    window:
        Rolling window length (must be ``>= 2``).

    Returns
    -------
    pandas.Series
        The causal return z-score, indexed like ``returns`` (leading warm-up
        rows are NaN).

    Raises
    ------
    ValidationError
        If ``returns`` is malformed or ``window < 2``.
    NotImplementedError
        Always — this is a typed stub for parallel authoring.
    """
    raise NotImplementedError
