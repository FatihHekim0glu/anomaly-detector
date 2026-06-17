"""Plotly figure builders for the anomaly-detector tool.

Each builder returns a plain ``dict`` shaped ``{"data": [...], "layout": {...}}``
— the same JSON shape the FastAPI layer serializes and the Next.js
``PlotlyChart`` component renders — so no Plotly object leaks across the API
boundary. Plotly is an OPTIONAL dependency (the ``viz`` extra) imported LAZILY
inside each builder; importing this module has no side effects and does not
require Plotly.

Two figures back the tool:

- :func:`price_anomaly_figure` — the price (and/or return) series with markers on
  the flagged anomalous days.
- :func:`score_threshold_figure` — the per-day anomaly-score series with the
  train-derived threshold drawn as a horizontal line.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

#: A Plotly figure serialized as a plain mapping with ``data`` and ``layout`` keys.
FigureDict = dict[str, Any]


def price_anomaly_figure(
    prices: pd.Series,
    flags: pd.Series,
    *,
    title: str = "Price with flagged anomalies",
) -> FigureDict:
    """Build a price line chart with markers on flagged anomalous days.

    Renders the price series as a line and overlays a scatter of markers at the
    dates where ``flags`` is ``True``, so the reader sees where each anomaly
    landed relative to the price path.

    Parameters
    ----------
    prices:
        The price (or return) level series, indexed by date.
    flags:
        Boolean per-day flag series aligned to ``prices``.
    title:
        The figure title.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping.

    Raises
    ------
    NotImplementedError
        Always — this is a typed stub for parallel authoring.
    """
    raise NotImplementedError


def score_threshold_figure(
    scores: pd.Series,
    threshold: float,
    *,
    title: str = "Anomaly score with threshold",
) -> FigureDict:
    """Build the per-day anomaly-score series with a horizontal threshold line.

    Renders ``scores`` as a line and draws the train-derived ``threshold`` as a
    horizontal reference line, so the reader sees which days breach it.

    Parameters
    ----------
    scores:
        The per-day anomaly score series over the OOS slice, indexed by date.
    threshold:
        The train-derived flag threshold to draw as a horizontal line.
    title:
        The figure title.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping.

    Raises
    ------
    NotImplementedError
        Always — this is a typed stub for parallel authoring.
    """
    raise NotImplementedError
