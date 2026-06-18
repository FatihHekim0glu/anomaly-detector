"""Unit coverage for the reused HRP infrastructure modules.

These exercise the leakage-guarded walk-forward engine, the performance/cost
helpers, the Probabilistic/Deflated Sharpe edge cases, the reproducibility
manifest, the seeded RNG substreams, the validation guardrails, and the shared
``AnomalyResult`` serialization - the infra the detectors and the scan stand on.

Everything is deterministic and network-free.
"""

from __future__ import annotations

import json
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from anomaly_detector._exceptions import (
    InsufficientDataError,
    NotFittedError,
    ValidationError,
)

# --------------------------------------------------------------------------- #
# AnomalyResult.to_dict / flagged_dates                                       #
# --------------------------------------------------------------------------- #


def _make_result(detector: str = "iforest") -> object:
    from anomaly_detector.detectors.result import AnomalyResult

    idx = pd.date_range("2021-01-04", periods=5, freq="B")
    scores = pd.Series([0.1, 0.9, 0.3, 0.8, 0.2], index=idx, name="anomaly_score")
    flags = pd.Series([False, True, False, True, False], index=idx, name="anomaly_flag")
    return AnomalyResult(
        scores=scores,
        flags=flags,
        threshold=0.5,
        detector=detector,
        contamination=0.04,
        n_train=60,
        n_test=5,
        meta={"feature_names": ["a", "b"]},
    )


def test_anomaly_result_to_dict_is_json_safe() -> None:
    result = _make_result()
    payload = result.to_dict()  # type: ignore[attr-defined]
    assert set(payload) == {
        "scores",
        "flags",
        "threshold",
        "detector",
        "contamination",
        "n_train",
        "n_test",
        "meta",
    }
    assert payload["threshold"] == 0.5
    assert payload["detector"] == "iforest"
    # ISO date keys, plain bool flags, finite-or-None scores.
    assert all(isinstance(k, str) for k in payload["scores"])
    assert all(isinstance(v, bool) for v in payload["flags"].values())
    json.loads(json.dumps(payload))


def test_anomaly_result_to_dict_scrubs_non_finite_scores() -> None:
    from anomaly_detector.detectors.result import AnomalyResult

    idx = pd.date_range("2021-01-04", periods=3, freq="B")
    scores = pd.Series([np.nan, np.inf, 0.5], index=idx)
    flags = pd.Series([False, False, True], index=idx)
    result = AnomalyResult(
        scores=scores,
        flags=flags,
        threshold=0.4,
        detector="autoencoder",
        contamination=0.02,
        n_train=10,
        n_test=3,
    )
    payload = result.to_dict()
    values = list(payload["scores"].values())
    assert values[0] is None and values[1] is None
    assert values[2] == 0.5


def test_anomaly_result_flagged_dates_sorted_by_score() -> None:
    result = _make_result()
    dates = result.flagged_dates()  # type: ignore[attr-defined]
    # Two flagged days; the higher-scoring one (0.9 at 2021-01-05) comes first.
    assert [d[:10] for d in dates] == ["2021-01-05", "2021-01-07"]


def test_anomaly_result_flagged_dates_empty_when_no_flags() -> None:
    from anomaly_detector.detectors.result import AnomalyResult

    idx = pd.date_range("2021-01-04", periods=3, freq="B")
    result = AnomalyResult(
        scores=pd.Series([0.1, 0.2, 0.3], index=idx),
        flags=pd.Series([False, False, False], index=idx),
        threshold=1.0,
        detector="iforest",
        contamination=0.02,
        n_train=5,
        n_test=3,
    )
    assert result.flagged_dates() == []


def test_iso_helper_handles_non_datetime_labels() -> None:
    from anomaly_detector.detectors.result import _iso

    assert _iso(pd.Timestamp("2020-01-01")).startswith("2020-01-01")
    assert _iso(7) == "7"
    assert _iso("label") == "label"


# --------------------------------------------------------------------------- #
# _rng substreams                                                             #
# --------------------------------------------------------------------------- #


def test_spawn_substreams_independent_and_reproducible() -> None:
    from anomaly_detector._rng import spawn_substreams

    a = spawn_substreams(123, 3)
    b = spawn_substreams(123, 3)
    assert len(a) == 3
    draws_a = [gen.standard_normal(4).tolist() for gen in a]
    draws_b = [gen.standard_normal(4).tolist() for gen in b]
    assert draws_a == draws_b  # reproducible
    assert draws_a[0] != draws_a[1]  # independent substreams differ


def test_spawn_substreams_rejects_negative() -> None:
    from anomaly_detector._rng import make_rng, spawn_substreams

    with pytest.raises(ValueError):
        spawn_substreams(-1, 2)
    with pytest.raises(ValueError):
        spawn_substreams(1, -2)
    with pytest.raises(ValueError):
        make_rng(-5)


# --------------------------------------------------------------------------- #
# _validation guardrails                                                      #
# --------------------------------------------------------------------------- #


def test_ensure_series_rejects_nan_and_2d_and_empty() -> None:
    from anomaly_detector._validation import ensure_series

    with pytest.raises(ValidationError):
        ensure_series(np.array([[1.0, 2.0]]))  # 2-D ndarray
    with pytest.raises(ValidationError):
        ensure_series(pd.Series([], dtype="float64"))  # empty
    with pytest.raises(ValidationError):
        ensure_series(pd.Series([1.0, np.nan]))  # NaN not allowed
    # allow_nan lets NaN through.
    out = ensure_series(pd.Series([1.0, np.nan]), allow_nan=True)
    assert bool(out.isna().any())


def test_ensure_dataframe_rejects_nan_and_3d_and_empty() -> None:
    from anomaly_detector._validation import ensure_dataframe

    with pytest.raises(ValidationError):
        ensure_dataframe(np.zeros((2, 2, 2)))  # 3-D
    with pytest.raises(ValidationError):
        ensure_dataframe(pd.DataFrame())  # empty
    with pytest.raises(ValidationError):
        ensure_dataframe(pd.DataFrame({"a": [1.0, np.nan]}))  # NaN
    frame = ensure_dataframe(np.array([[1.0, 2.0]]), columns=["x", "y"])
    assert list(frame.columns) == ["x", "y"]


def test_align_inner_and_min_obs() -> None:
    from anomaly_detector._validation import align_inner, validate_min_obs

    left = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=[0, 1, 2])
    right = pd.DataFrame({"b": [9.0, 8.0]}, index=[1, 2])
    la, ra = align_inner(left, right)
    assert list(la.index) == [1, 2]
    assert list(ra.index) == [1, 2]

    with pytest.raises(ValidationError):
        align_inner(left, pd.DataFrame({"b": [1.0]}, index=[99]))

    with pytest.raises(InsufficientDataError):
        validate_min_obs(left, 10)
    validate_min_obs(left, 3)  # exactly enough: no raise


# --------------------------------------------------------------------------- #
# backtest/stats + costs                                                      #
# --------------------------------------------------------------------------- #


def test_sharpe_and_vol_and_drawdown() -> None:
    from anomaly_detector.backtest.stats import (
        annualized_vol,
        max_drawdown,
        sharpe_ratio,
    )

    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.0005, 0.01, 500))
    sr = sharpe_ratio(rets)
    assert math.isfinite(sr)
    assert annualized_vol(rets) > 0.0
    # A flat series has undefined Sharpe.
    assert math.isnan(sharpe_ratio(pd.Series([0.0] * 10)))
    # Drawdown is non-positive; a monotonically rising series has zero drawdown.
    assert max_drawdown(pd.Series([0.01, 0.01, 0.01])) == 0.0
    assert max_drawdown(pd.Series([0.1, -0.5, 0.05])) < 0.0


def test_turnover_aligns_and_halves() -> None:
    from anomaly_detector.backtest.stats import turnover

    prev = pd.Series({"A": 1.0, "B": 0.0})
    new = pd.Series({"A": 0.0, "C": 1.0})
    # Full rotation A->C across the union {A,B,C}: |-1|+0+|1| = 2, halved => 1.0.
    assert turnover(prev, new) == pytest.approx(1.0)


def test_fixed_bps_cost_validation_and_value() -> None:
    from anomaly_detector.backtest.costs import FixedBpsCost

    cost = FixedBpsCost(bps=10.0)
    assert cost.cost(0.5) == pytest.approx(0.5 * 10.0 / 10_000.0)
    with pytest.raises(ValidationError):
        FixedBpsCost(bps=-1.0)
    with pytest.raises(ValidationError):
        cost.cost(-0.1)


# --------------------------------------------------------------------------- #
# walk_forward_backtest                                                       #
# --------------------------------------------------------------------------- #


def _equal_weight(window: pd.DataFrame) -> pd.Series:
    cols = list(window.columns)
    return pd.Series([1.0 / len(cols)] * len(cols), index=cols)


def test_walk_forward_runs_and_costs_are_monotone() -> None:
    from anomaly_detector.backtest.walk_forward import walk_forward_backtest

    rng = np.random.default_rng(1)
    idx = pd.date_range("2015-01-01", periods=400, freq="B")
    panel = pd.DataFrame(rng.normal(0.0003, 0.01, (400, 3)), index=idx, columns=["A", "B", "C"])

    res_lo = walk_forward_backtest(
        panel, _equal_weight, lookback_window=63, rebalance="monthly", cost_bps=0.0
    )
    res_hi = walk_forward_backtest(
        panel, _equal_weight, lookback_window=63, rebalance="monthly", cost_bps=50.0
    )
    assert res_lo.n_rebalances > 0
    # Higher cost => lower (or equal) net total return.
    assert res_hi.oos_returns.sum() <= res_lo.oos_returns.sum() + 1e-9
    # Result serializes cleanly.
    json.loads(json.dumps(res_lo.to_dict()))


def test_walk_forward_anchored_and_quarterly() -> None:
    from anomaly_detector.backtest.walk_forward import walk_forward_backtest

    rng = np.random.default_rng(2)
    idx = pd.date_range("2015-01-01", periods=500, freq="B")
    panel = pd.DataFrame(rng.normal(0.0, 0.01, (500, 2)), index=idx, columns=["A", "B"])
    res = walk_forward_backtest(
        panel, _equal_weight, lookback_window=63, rebalance="quarterly", anchored=True
    )
    assert res.n_rebalances >= 1
    assert res.meta["anchored"] is True


def test_walk_forward_validation_errors() -> None:
    from anomaly_detector.backtest.walk_forward import walk_forward_backtest

    idx = pd.date_range("2015-01-01", periods=200, freq="B")
    panel = pd.DataFrame(np.zeros((200, 2)), index=idx, columns=["A", "B"])

    with pytest.raises(ValidationError):
        walk_forward_backtest(panel, _equal_weight, lookback_window=63, cost_bps=-1.0)
    with pytest.raises(ValidationError):
        walk_forward_backtest(panel, _equal_weight, lookback_window=63, rebalance="weekly")
    with pytest.raises(ValidationError):
        walk_forward_backtest(panel, _equal_weight, lookback_window=2)  # < n_assets + 1
    with pytest.raises(ValidationError):
        walk_forward_backtest(panel, _equal_weight, lookback_window=63, purge=-1)


def test_walk_forward_insufficient_data() -> None:
    from anomaly_detector.backtest.walk_forward import walk_forward_backtest

    idx = pd.date_range("2015-01-01", periods=40, freq="B")
    panel = pd.DataFrame(np.zeros((40, 2)), index=idx, columns=["A", "B"])
    with pytest.raises(InsufficientDataError):
        walk_forward_backtest(panel, _equal_weight, lookback_window=63)


# --------------------------------------------------------------------------- #
# dsr edge cases                                                              #
# --------------------------------------------------------------------------- #


def test_psr_and_dsr_basic_and_monotone() -> None:
    from anomaly_detector.evaluation.dsr import (
        deflated_sharpe_ratio,
        probabilistic_sharpe_ratio,
    )

    psr = probabilistic_sharpe_ratio(0.1, n_obs=252, skew=-0.5, kurtosis=4.0)
    assert 0.0 <= psr <= 1.0

    # The DSR is non-increasing in the number of trials (multiplicity penalty).
    common = {"n_obs": 252, "variance_of_trial_sharpes": 0.02}
    dsr_few = deflated_sharpe_ratio(0.15, n_trials=1, **common)
    dsr_many = deflated_sharpe_ratio(0.15, n_trials=100, **common)
    assert dsr_many <= dsr_few

    # n_trials == 1 collapses the benchmark to zero (DSR == PSR vs 0).
    dsr_one = deflated_sharpe_ratio(0.15, n_trials=1, n_obs=252, variance_of_trial_sharpes=0.02)
    psr_zero = probabilistic_sharpe_ratio(0.15, n_obs=252)
    assert dsr_one == pytest.approx(psr_zero, abs=1e-12)


def test_dsr_validation_errors() -> None:
    from anomaly_detector.evaluation.dsr import (
        deflated_sharpe_ratio,
        probabilistic_sharpe_ratio,
    )

    with pytest.raises(ValidationError):
        probabilistic_sharpe_ratio(0.1, n_obs=1)
    with pytest.raises(ValidationError):
        deflated_sharpe_ratio(0.1, n_obs=1, n_trials=10, variance_of_trial_sharpes=0.1)
    with pytest.raises(ValidationError):
        deflated_sharpe_ratio(0.1, n_obs=252, n_trials=0, variance_of_trial_sharpes=0.1)
    with pytest.raises(ValidationError):
        deflated_sharpe_ratio(0.1, n_obs=252, n_trials=5, variance_of_trial_sharpes=-0.1)
    # A degenerate (non-positive) variance term is rejected. With kurtosis == 1
    # the quadratic term vanishes and a large skew*SR drives the bracket
    # variance (1 - skew*SR) negative.
    with pytest.raises(ValidationError):
        probabilistic_sharpe_ratio(1.0, n_obs=10, skew=2.5, kurtosis=1.0)


def test_norm_ppf_round_trips_through_cdf() -> None:
    from anomaly_detector.evaluation.dsr import _norm_cdf, _norm_ppf

    for p in (0.01, 0.2, 0.5, 0.8, 0.99):
        assert _norm_cdf(_norm_ppf(p)) == pytest.approx(p, abs=1e-9)
    with pytest.raises(ValidationError):
        _norm_ppf(0.0)


# --------------------------------------------------------------------------- #
# _manifest                                                                   #
# --------------------------------------------------------------------------- #


def test_config_hash_is_order_independent_and_stable() -> None:
    from anomaly_detector._manifest import config_hash

    h1 = config_hash({"a": 1, "b": 2})
    h2 = config_hash({"b": 2, "a": 1})
    assert h1 == h2
    assert len(h1) == 32
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_run_manifest_capture_and_to_dict() -> None:
    from anomaly_detector._manifest import RunManifest

    manifest = RunManifest.capture({"ticker": "SPY", "seed": 7}, seed=7)
    payload = manifest.to_dict()
    assert set(payload) >= {"git_sha", "dirty", "config_hash", "seed"}
    assert payload["seed"] == 7
    assert isinstance(payload["dirty"], bool)
    json.loads(json.dumps(payload))


# --------------------------------------------------------------------------- #
# PolygonProvider internals (no network)                                      #
# --------------------------------------------------------------------------- #


def test_polygon_series_from_payload_parses_bars() -> None:
    from anomaly_detector.data_providers.polygon import PolygonProvider

    payload = {
        "status": "OK",
        "results": [
            {"t": 1_577_836_800_000, "c": 100.0},  # 2020-01-01
            {"t": 1_577_923_200_000, "c": 101.0},  # 2020-01-02
        ],
    }
    series = PolygonProvider._series_from_payload(payload, "SPY")
    assert list(series.to_numpy()) == [100.0, 101.0]
    assert series.name == "SPY"


def test_polygon_series_from_payload_empty_raises() -> None:
    from anomaly_detector.data_providers.polygon import PolygonProvider

    with pytest.raises(ValueError):
        PolygonProvider._series_from_payload({"status": "NOT_FOUND", "results": []}, "SPY")


def test_polygon_fetch_uses_monkeypatched_json(monkeypatch: pytest.MonkeyPatch) -> None:
    from anomaly_detector.data_providers import polygon as polygon_mod

    provider = polygon_mod.PolygonProvider(api_key="dummy")

    def _fake_get_json(self: object, ticker: str, start: date, end: date) -> dict[str, object]:
        return {
            "status": "OK",
            "results": [
                {"t": 1_577_836_800_000, "c": 50.0},
                {"t": 1_577_923_200_000, "c": 51.0},
            ],
        }

    monkeypatch.setattr(polygon_mod.PolygonProvider, "_get_json", _fake_get_json)
    frame = provider.fetch(["SPY"], date(2020, 1, 1), date(2020, 1, 3))
    assert list(frame.columns) == ["SPY"]
    assert frame.shape[0] == 2


def test_polygon_fetch_validation_errors() -> None:
    from anomaly_detector.data_providers.polygon import PolygonProvider

    provider = PolygonProvider(api_key="dummy")
    with pytest.raises(ValidationError):
        provider.fetch([], date(2020, 1, 1), date(2020, 2, 1))
    with pytest.raises(ValidationError):
        provider.fetch(["SPY"], date(2020, 2, 1), date(2020, 1, 1))


def test_polygon_url_includes_ticker_and_key() -> None:
    from anomaly_detector.data_providers.polygon import PolygonProvider

    provider = PolygonProvider(api_key="SECRET")
    url = provider._url("SPY", date(2020, 1, 1), date(2020, 2, 1))
    assert "SPY" in url
    assert "SECRET" in url
    assert "2020-01-01" in url and "2020-02-01" in url


def test_polygon_resolve_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from anomaly_detector.data_providers import polygon as polygon_mod

    monkeypatch.setenv("POLYGON_API_KEY", "ENVKEY")
    assert polygon_mod._resolve_api_key(None) == "ENVKEY"
    assert polygon_mod._resolve_api_key("EXPLICIT") == "EXPLICIT"


def test_polygon_resolve_api_key_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from anomaly_detector.data_providers import polygon as polygon_mod

    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    monkeypatch.setattr(polygon_mod, "_load_api_key_from_dotenv", lambda: None)
    with pytest.raises(ValidationError):
        polygon_mod._resolve_api_key(None)


def test_iforest_score_before_fit_raises_notfitted() -> None:
    from anomaly_detector.detectors.iforest import IsolationForestDetector

    det = IsolationForestDetector()
    with pytest.raises(NotFittedError):
        det.score(pd.DataFrame({"a": [0.1], "b": [0.2]}))
