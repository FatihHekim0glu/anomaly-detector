"""Unit coverage for the public ``run_anomaly_scan`` orchestration surface.

These run on the seeded synthetic fixture with NO network, so they cover the
walk-forward orchestration in :mod:`anomaly_detector.scan` under the default
(non-integration) test selection that CI uses. They check the summary shape and
JSON-safety, the price/return input parity, the figure builders, and the
validation/short-series error paths.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from anomaly_detector import run_anomaly_scan
from anomaly_detector._exceptions import InsufficientDataError, ValidationError

pytestmark = pytest.mark.unit

_SUMMARY_KEYS = {
    "n_flags",
    "jaccard",
    "proxy_precision",
    "proxy_recall",
    "top_anomaly_dates",
    "detector",
    "data_source",
}


@pytest.mark.parametrize("detector", ["iforest", "autoencoder", "both"])
def test_scan_summary_is_json_safe_for_each_detector(injected_anomalies, detector: str) -> None:
    """Every detector choice yields a JSON-safe summary with bounded ratios."""
    result = run_anomaly_scan(
        prices=injected_anomalies.prices,
        detector=detector,
        contamination=0.02,
        window=21,
        seed=7,
        data_source="synthetic",
    )
    summary = result.summary()

    assert set(summary) == _SUMMARY_KEYS
    assert summary["detector"] == detector
    assert summary["data_source"] == "synthetic"
    assert summary["n_flags"] >= 0
    assert isinstance(summary["top_anomaly_dates"], list)
    for key in ("jaccard", "proxy_precision", "proxy_recall"):
        assert 0.0 <= summary[key] <= 1.0
    json.loads(json.dumps(summary))


def test_scan_provenance_meta_and_shared_oos_index(injected_anomalies) -> None:
    """Both detectors score the same disjoint OOS index under a walk-forward."""
    result = run_anomaly_scan(prices=injected_anomalies.prices, seed=7)

    assert result.meta["n_folds"] >= 1
    assert result.meta["n_oos"] == result.result_iforest.n_test
    assert result.result_iforest.meta["walk_forward"] is True
    assert result.result_autoencoder.meta["walk_forward"] is True
    assert result.result_iforest.scores.index.equals(result.result_autoencoder.scores.index)
    assert result.oos_returns.index.equals(result.result_iforest.scores.index)

    json.loads(json.dumps(result.result_iforest.to_dict()))
    json.loads(json.dumps(result.agreement.to_dict()))


def test_scan_returns_input_matches_price_input(injected_anomalies) -> None:
    """A return-series input integrates to a price path and scans identically."""
    from_returns = run_anomaly_scan(returns=injected_anomalies.returns, detector="iforest", seed=7)
    from_prices = run_anomaly_scan(prices=injected_anomalies.prices, detector="iforest", seed=7)

    assert from_returns.summary()["data_source"] is None
    np.testing.assert_allclose(
        from_returns.result_iforest.scores.to_numpy(),
        from_prices.result_iforest.scores.to_numpy(),
        atol=1e-9,
        rtol=0.0,
    )


def test_scan_figures_are_json_clean(injected_anomalies) -> None:
    """The price/score figures serialize to JSON-clean ``{data, layout}`` dicts."""
    result = run_anomaly_scan(prices=injected_anomalies.prices, detector="both", seed=7)
    figures = result.figures()

    assert set(figures) == {"price_figure", "score_figure"}
    for fig in figures.values():
        assert set(fig) == {"data", "layout"}
        assert isinstance(fig["data"], list)
    json.loads(json.dumps(figures))


def test_scan_rejects_both_or_neither_input(injected_anomalies) -> None:
    """Exactly one of prices/returns is required."""
    with pytest.raises(ValidationError):
        run_anomaly_scan()
    with pytest.raises(ValidationError):
        run_anomaly_scan(prices=injected_anomalies.prices, returns=injected_anomalies.returns)


def test_scan_rejects_out_of_range_parameters(injected_anomalies) -> None:
    """An unknown detector or out-of-range contamination is rejected."""
    with pytest.raises(ValidationError):
        run_anomaly_scan(prices=injected_anomalies.prices, detector="bogus")
    with pytest.raises(ValidationError):
        run_anomaly_scan(prices=injected_anomalies.prices, contamination=0.9)


def test_scan_too_short_series_raises() -> None:
    """A series too short to carve a walk-forward fold raises clearly."""
    idx = pd.date_range("2020-01-01", periods=30, freq="B")
    prices = pd.Series(100.0 + np.arange(30, dtype="float64"), index=idx, name="price")
    with pytest.raises(InsufficientDataError):
        run_anomaly_scan(prices=prices, window=21)


def test_scan_rejects_multi_column_price_frame(injected_anomalies) -> None:
    """A price DataFrame with more than one column is rejected."""
    frame = pd.DataFrame(
        {
            "a": injected_anomalies.prices.to_numpy(),
            "b": injected_anomalies.prices.to_numpy(),
        },
        index=injected_anomalies.prices.index,
    )
    with pytest.raises(ValidationError):
        run_anomaly_scan(prices=frame)
