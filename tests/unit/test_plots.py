"""Tests for the lazy-Plotly figure builders.

Asserts the two figure builders return valid ``{data, layout}`` mappings with
finite, JSON-serializable contents, place markers at exactly the flagged indices,
and draw the threshold line — and that importing :mod:`anomaly_detector.plots`
pulls in no Plotly at module-import time (lazy-import discipline).
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from anomaly_detector.plots import (
    _jsonify,
    price_anomaly_figure,
    score_threshold_figure,
)


def _assert_json_finite(obj: Any) -> None:
    """Assert ``obj`` JSON-serializes and contains no non-finite floats."""
    text = json.dumps(obj)  # raises if any non-serializable object leaks through
    assert isinstance(text, str)

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for value in node:
                _walk(value)
        elif isinstance(node, float):
            assert math.isfinite(node), "non-finite float leaked into figure"

    _walk(obj)


@pytest.fixture
def _series_with_flags() -> tuple[pd.Series, pd.Series]:
    """A 12-point price series and a boolean flag series with two True days."""
    index = pd.date_range("2020-01-01", periods=12, freq="B")
    prices = pd.Series(np.linspace(100.0, 111.0, 12), index=index, name="price")
    flags = pd.Series(False, index=index, name="anomaly_flag")
    flags.iloc[4] = True
    flags.iloc[9] = True
    return prices, flags


@pytest.mark.unit
def test_price_figure_has_data_and_layout(_series_with_flags: tuple[pd.Series, pd.Series]) -> None:
    """The price figure is a ``{data, layout}`` mapping that JSON-serializes."""
    prices, flags = _series_with_flags
    fig = price_anomaly_figure(prices, flags)

    assert set(fig) == {"data", "layout"}
    assert isinstance(fig["data"], list) and len(fig["data"]) >= 2
    assert isinstance(fig["layout"], dict)
    assert fig["layout"]["title"]["text"] == "Price with flagged anomalies"
    _assert_json_finite(fig)


@pytest.mark.unit
def test_price_figure_markers_at_flagged_indices(
    _series_with_flags: tuple[pd.Series, pd.Series],
) -> None:
    """Anomaly markers land on exactly the flagged dates (and their prices)."""
    prices, flags = _series_with_flags
    fig = price_anomaly_figure(prices, flags)

    marker_trace = next(t for t in fig["data"] if t.get("mode") == "markers")
    expected_dates = [prices.index[4].isoformat(), prices.index[9].isoformat()]
    assert list(marker_trace["x"]) == expected_dates
    # The marker y-values are the prices on those flagged days.
    assert list(marker_trace["y"]) == [
        float(prices.iloc[4]),
        float(prices.iloc[9]),
    ]


@pytest.mark.unit
def test_price_figure_no_flags_yields_empty_marker_trace() -> None:
    """With no flags the marker trace is present but empty (still valid)."""
    index = pd.date_range("2021-01-01", periods=6, freq="B")
    prices = pd.Series(np.arange(6, dtype="float64") + 100.0, index=index)
    flags = pd.Series(False, index=index)

    fig = price_anomaly_figure(prices, flags, title="Custom title")
    marker_trace = next(t for t in fig["data"] if t.get("mode") == "markers")
    assert list(marker_trace["x"]) == []
    assert fig["layout"]["title"]["text"] == "Custom title"
    _assert_json_finite(fig)


@pytest.mark.unit
def test_price_figure_aligns_misindexed_flags() -> None:
    """Flags indexed on only a sub-slice mark only their own dates."""
    index = pd.date_range("2022-01-03", periods=8, freq="B")
    prices = pd.Series(np.linspace(50.0, 60.0, 8), index=index)
    # Flags cover only the back half (the OOS slice), with one True.
    oos_index = index[4:]
    flags = pd.Series(False, index=oos_index)
    flags.iloc[1] = True  # -> index[5]

    fig = price_anomaly_figure(prices, flags)
    marker_trace = next(t for t in fig["data"] if t.get("mode") == "markers")
    assert list(marker_trace["x"]) == [index[5].isoformat()]


@pytest.mark.unit
def test_score_figure_has_threshold_line_and_breach_markers() -> None:
    """The score figure draws a horizontal threshold line and marks breaches."""
    index = pd.date_range("2020-06-01", periods=10, freq="B")
    scores = pd.Series(
        np.array([0.1, 0.2, 5.0, 0.3, 0.1, 4.0, 0.2, 0.1, 0.05, 0.3]),
        index=index,
        name="anomaly_score",
    )
    threshold = 1.0

    fig = score_threshold_figure(scores, threshold)
    assert set(fig) == {"data", "layout"}

    # The horizontal threshold line is a shape spanning paper-x at y == threshold.
    line_shapes = [s for s in fig["layout"]["shapes"] if s["type"] == "line"]
    assert line_shapes, "expected a threshold line shape"
    assert line_shapes[0]["y0"] == pytest.approx(threshold)
    assert line_shapes[0]["y1"] == pytest.approx(threshold)

    # Breach markers land on exactly the days whose score exceeds the threshold.
    marker_trace = next(t for t in fig["data"] if t.get("mode") == "markers")
    expected = [index[2].isoformat(), index[5].isoformat()]
    assert list(marker_trace["x"]) == expected
    _assert_json_finite(fig)


@pytest.mark.unit
def test_score_figure_threshold_annotation_and_title() -> None:
    """The threshold value is annotated and the title is honoured."""
    index = pd.date_range("2021-03-01", periods=5, freq="B")
    scores = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=index)

    fig = score_threshold_figure(scores, 0.35, title="My scores")
    assert fig["layout"]["title"]["text"] == "My scores"
    annotations = fig["layout"]["annotations"]
    assert any("threshold" in a["text"] for a in annotations)


@pytest.mark.unit
def test_jsonify_maps_non_finite_floats_to_none() -> None:
    """Non-finite floats (NaN/Inf) become ``None`` so the figure stays JSON-safe."""
    assert _jsonify(float("nan")) is None
    assert _jsonify(float("inf")) is None
    assert _jsonify(float("-inf")) is None
    assert _jsonify(1.25) == 1.25


@pytest.mark.unit
def test_jsonify_handles_numpy_pandas_and_nested_scalars() -> None:
    """numpy arrays/scalars and pandas timestamps coerce to native JSON types."""
    arr = _jsonify(np.array([1.0, np.nan, 3.0]))
    assert arr == [1.0, None, 3.0]
    assert _jsonify(np.float64(2.5)) == 2.5
    ts = pd.Timestamp("2020-03-16")
    assert _jsonify(ts) == ts.isoformat()
    # Nested containers recurse.
    assert _jsonify({"a": (np.int64(7), float("inf"))}) == {"a": [7, None]}


@pytest.mark.unit
def test_figures_drop_non_finite_score_values() -> None:
    """A NaN score does not leak a non-finite float into the score figure."""
    index = pd.date_range("2020-01-01", periods=5, freq="B")
    scores = pd.Series([0.1, float("nan"), 0.3, 0.4, 0.5], index=index)
    fig = score_threshold_figure(scores, 0.35)
    _assert_json_finite(fig)


@pytest.mark.unit
def test_plots_module_import_is_plotly_free() -> None:
    """Importing the plots module does not eagerly import Plotly."""
    import importlib
    import sys

    # Drop any prior import so we observe this module's own import behaviour.
    sys.modules.pop("plotly", None)
    sys.modules.pop("anomaly_detector.plots", None)
    importlib.import_module("anomaly_detector.plots")
    assert "plotly" not in sys.modules, "plots.py must import Plotly lazily, not at import time"
