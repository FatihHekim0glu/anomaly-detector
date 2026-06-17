"""Optional "fade-the-anomaly" toy overlay (clearly labeled DIAGNOSTIC).

A deliberately simple, honest-null demonstration: on the day AFTER a detector
flags an anomaly, take a small mean-reverting ("fade") position, hold one day,
and measure the out-of-sample return. This is NOT a tradable strategy and makes
NO alpha claim — it exists only to put a Deflated-Sharpe number on the question
"do the flags carry any next-day information?", with the multiple-testing
penalty counting the FULL grid.

The Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014, reused from
:mod:`anomaly_detector.evaluation.dsr`) uses::

    n_trials = #detectors x #contamination-levels x #windows

so the honest yardstick already pays for every configuration explored. The
expected verdict is "indistinguishable from zero after deflation".

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True, slots=True)
class OverlayResult:
    """Immutable result of the toy fade-the-anomaly overlay.

    Attributes
    ----------
    oos_returns:
        The net next-day overlay return series (zero on non-flagged days),
        indexed by date.
    sharpe:
        The annualized Sharpe ratio of ``oos_returns`` (descriptive only).
    deflated_sharpe:
        The Deflated Sharpe Ratio against the FULL configuration grid; the
        honest, multiplicity-aware yardstick.
    n_trials:
        The multiplicity count fed to the DSR
        (``#detectors x #contamination x #windows``).
    n_trades:
        The number of non-zero (flagged) overlay days.
    meta:
        Free-form JSON-serializable provenance.
    """

    oos_returns: pd.Series
    sharpe: float
    deflated_sharpe: float
    n_trials: int
    n_trades: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result.

        Raises
        ------
        NotImplementedError
            Always — this is a typed stub for parallel authoring.
        """
        raise NotImplementedError


def fade_the_anomaly_overlay(
    flags: pd.Series,
    returns: pd.Series,
    *,
    n_trials: int = 1,
    cost_bps: float = 5.0,
) -> OverlayResult:
    """Run the toy fade-the-anomaly overlay and deflate its Sharpe.

    On each day flagged anomalous, the overlay takes a one-day fade position in
    the NEXT day's return (``flags.shift(1)`` so no flag is acted on before it is
    known), charges a per-side cost, and the realized series is summarized by a
    Deflated Sharpe Ratio counting ``n_trials`` configurations.

    DIAGNOSTIC ONLY: this is not a strategy and claims no alpha; the expected
    result is a DSR indistinguishable from zero.

    Parameters
    ----------
    flags:
        Boolean per-day anomaly flags over the OOS slice.
    returns:
        The per-day return series over the same OOS slice.
    n_trials:
        The FULL multiplicity count for the DSR
        (``#detectors x #contamination x #windows``).
    cost_bps:
        Per-side transaction cost in basis points.

    Returns
    -------
    OverlayResult
        The overlay's net return series with its descriptive and deflated Sharpe.

    Raises
    ------
    NotImplementedError
        Always — this is a typed stub for parallel authoring.
    """
    raise NotImplementedError
