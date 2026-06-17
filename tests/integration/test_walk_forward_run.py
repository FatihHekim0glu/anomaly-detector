"""Integration test: full causal walk-forward scan on the synthetic fixture.

Exercises the public :func:`anomaly_detector.run_anomaly_scan` entrypoint end to
end with NO network: the injected fixture -> causal features -> anchored
walk-forward refit (train-only per fold) -> disjoint OOS scoring -> descriptive
agreement summary -> JSON-clean serialization of every result and figure.

This is the integration coverage the brief requires on ``injected_anomalies``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from anomaly_detector import run_anomaly_scan
from anomaly_detector._exceptions import InsufficientDataError, ValidationError

pytestmark = pytest.mark.integration


def test_end_to_end_walk_forward_on_synthetic_fixture(injected_anomalies) -> None:
    """A full causal walk-forward scan produces a JSON-serializable summary."""
    result = run_anomaly_scan(
        prices=injected_anomalies.prices,
        detector="both",
        contamination=0.02,
        window=21,
        seed=7,
        data_source="synthetic",
    )

    summary = result.summary()

    # --- Summary shape and JSON-safety ------------------------------------
    assert set(summary) == {
        "n_flags",
        "jaccard",
        "proxy_precision",
        "proxy_recall",
        "top_anomaly_dates",
        "detector",
        "data_source",
    }
    assert summary["detector"] == "both"
    assert summary["data_source"] == "synthetic"
    assert summary["n_flags"] > 0
    assert isinstance(summary["top_anomaly_dates"], list)
    assert all(isinstance(d, str) for d in summary["top_anomaly_dates"])
    for key in ("jaccard", "proxy_precision", "proxy_recall"):
        assert 0.0 <= summary[key] <= 1.0
    # The whole summary round-trips through JSON without error.
    json.loads(json.dumps(summary))

    # --- Walk-forward provenance ------------------------------------------
    assert result.meta["n_folds"] >= 1
    assert result.meta["n_oos"] == result.result_iforest.n_test
    assert result.result_iforest.meta["walk_forward"] is True
    assert result.result_autoencoder.meta["walk_forward"] is True

    # --- Both detectors scored the SAME disjoint OOS index ----------------
    assert result.result_iforest.scores.index.equals(result.result_autoencoder.scores.index)
    assert result.oos_returns.index.equals(result.result_iforest.scores.index)

    # --- Every result serializes cleanly ----------------------------------
    json.loads(json.dumps(result.result_iforest.to_dict()))
    json.loads(json.dumps(result.result_autoencoder.to_dict()))
    json.loads(json.dumps(result.agreement.to_dict()))


def test_scan_no_lookahead_oos_scores_are_walk_forward(injected_anomalies) -> None:
    """Mutating bars after the final OOS day cannot change earlier OOS scores.

    The whole point of the walk-forward refit is that each OOS day's score uses
    only information available before it. Perturbing the tail of the price path
    must leave every score before the perturbation untouched.
    """
    prices = injected_anomalies.prices

    base = run_anomaly_scan(prices=prices, seed=7).result_iforest.scores

    perturbed_prices = prices.copy()
    perturbed_prices.iloc[-20:] = perturbed_prices.iloc[-20:] * 1.5
    perturbed = run_anomaly_scan(prices=perturbed_prices, seed=7).result_iforest.scores

    # Compare on the shared early dates (before the perturbation window).
    shared = base.index.intersection(perturbed.index)[:-40]
    np.testing.assert_allclose(
        base.reindex(shared).to_numpy(),
        perturbed.reindex(shared).to_numpy(),
        atol=1e-9,
        rtol=0.0,
    )


def test_scan_figures_are_json_safe(injected_anomalies) -> None:
    """The price/score figures serialize to JSON-clean ``{data, layout}`` dicts."""
    result = run_anomaly_scan(prices=injected_anomalies.prices, detector="iforest", seed=7)
    figures = result.figures()

    assert set(figures) == {"price_figure", "score_figure"}
    for fig in figures.values():
        assert set(fig) == {"data", "layout"}
        assert isinstance(fig["data"], list)
    json.loads(json.dumps(figures))


def test_scan_returns_input_path(injected_anomalies) -> None:
    """A return-series input is integrated to a price path and scanned identically."""
    result = run_anomaly_scan(returns=injected_anomalies.returns, detector="autoencoder", seed=7)
    summary = result.summary()
    assert summary["detector"] == "autoencoder"
    assert summary["n_flags"] >= 0
    assert summary["data_source"] is None


def test_scan_rejects_both_or_neither_inputs(injected_anomalies) -> None:
    """Exactly one of prices/returns is required."""
    with pytest.raises(ValidationError):
        run_anomaly_scan()
    with pytest.raises(ValidationError):
        run_anomaly_scan(prices=injected_anomalies.prices, returns=injected_anomalies.returns)


def test_scan_rejects_bad_parameters(injected_anomalies) -> None:
    """Out-of-range detector/contamination are rejected."""
    with pytest.raises(ValidationError):
        run_anomaly_scan(prices=injected_anomalies.prices, detector="bogus")
    with pytest.raises(ValidationError):
        run_anomaly_scan(prices=injected_anomalies.prices, contamination=0.9)


def test_scan_too_short_series_raises() -> None:
    """A series too short to carve a walk-forward fold raises clearly."""
    import pandas as pd

    idx = pd.date_range("2020-01-01", periods=30, freq="B")
    prices = pd.Series(100.0 + np.arange(30, dtype="float64"), index=idx, name="price")
    with pytest.raises(InsufficientDataError):
        run_anomaly_scan(prices=prices, window=21)
