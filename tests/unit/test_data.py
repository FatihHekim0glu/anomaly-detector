"""Unit tests for the data layer (``anomaly_detector.data``).

Covers the synthetic anomaly-injected generator, the offline-degrading price
loader, and the no-lookahead return helper:

- generator determinism (same seed -> byte-identical series; different seed ->
  different series) and validation of its size guards;
- injected anomalies are RECORDED and recoverable: the injected vol bursts and
  jumps actually inflate the realized return magnitude at the known indices,
  and ``known_anomaly_idx`` expands burst windows correctly;
- the loader degrades to the deterministic synthetic path offline (no network)
  and honours ``source_pref="synthetic"`` / cache flags;
- ``compute_returns`` differences with ``pct_change(fill_method=None)`` (never
  ffill-then-diff) across all accepted input shapes;
- import purity: importing the module touches no network and no heavy deps.

No test here touches the network.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from anomaly_detector._exceptions import ValidationError
from anomaly_detector.data import (
    DEFAULT_TICKER,
    InjectedSeries,
    compute_returns,
    generate_injected_series,
    load_prices,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Generator: determinism                                                      #
# --------------------------------------------------------------------------- #
def test_generator_is_deterministic_same_seed() -> None:
    """Same seed -> byte-identical returns, prices, and injection indices."""
    a = generate_injected_series(n_obs=400, seed=7)
    b = generate_injected_series(n_obs=400, seed=7)
    pd.testing.assert_series_equal(a.returns, b.returns)
    pd.testing.assert_series_equal(a.prices, b.prices)
    assert a.vol_burst_idx == b.vol_burst_idx
    assert a.jump_idx == b.jump_idx
    assert a.meta == b.meta


def test_generator_differs_across_seeds() -> None:
    """Different seeds -> different return paths (but same shape)."""
    a = generate_injected_series(n_obs=400, seed=7)
    b = generate_injected_series(n_obs=400, seed=8)
    assert a.returns.shape == b.returns.shape
    assert not np.allclose(a.returns.to_numpy(), b.returns.to_numpy())


def test_generator_shapes_and_index() -> None:
    """The generator returns aligned, business-day-indexed return/price Series."""
    s = generate_injected_series(n_obs=300, seed=1, start=date(2018, 3, 1))
    assert isinstance(s, InjectedSeries)
    assert len(s.returns) == 300
    assert len(s.prices) == 300
    assert isinstance(s.returns.index, pd.DatetimeIndex)
    assert s.returns.index.equals(s.prices.index)
    assert s.returns.notna().all()
    assert (s.prices > 0).all()  # strictly positive price path
    assert s.returns.index[0] == pd.Timestamp(date(2018, 3, 1))


# --------------------------------------------------------------------------- #
# Generator: validation                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_obs": 0},
        {"n_obs": -5},
        {"n_obs": 20},  # too small for 6 bursts + 6 jumps
        {"base_vol": 0.0},
        {"base_vol": -0.01},
        {"burst_vol_mult": 0.0},
        {"jump_size": -0.01},
        {"n_vol_bursts": -1},
        {"n_jumps": -2},
    ],
)
def test_generator_rejects_bad_params(kwargs: dict[str, object]) -> None:
    """Non-positive sizes and an under-sized series raise ``ValidationError``."""
    with pytest.raises(ValidationError):
        generate_injected_series(**kwargs)  # type: ignore[arg-type]


def test_generator_allows_zero_injections() -> None:
    """Zero bursts and zero jumps is valid (a calm null series)."""
    s = generate_injected_series(n_obs=200, seed=3, n_vol_bursts=0, n_jumps=0)
    assert s.vol_burst_idx == ()
    assert s.jump_idx == ()
    assert s.known_anomaly_idx() == ()


# --------------------------------------------------------------------------- #
# Injected anomalies: recoverability + bookkeeping                            #
# --------------------------------------------------------------------------- #
def test_injection_indices_are_recorded_and_in_back_half() -> None:
    """The injection indices are recorded and live in the back half of the series."""
    n = 1000
    s = generate_injected_series(n_obs=n, seed=7, n_vol_bursts=6, n_jumps=6)
    assert len(s.vol_burst_idx) == 6
    assert len(s.jump_idx) == 6
    for i in (*s.vol_burst_idx, *s.jump_idx):
        assert n // 2 <= i < n  # back half, so a causal split keeps them OOS


def test_known_anomaly_idx_expands_burst_windows() -> None:
    """``known_anomaly_idx`` expands each 5-day burst window and de-duplicates."""
    s = generate_injected_series(n_obs=600, seed=11)
    known = s.known_anomaly_idx()
    assert list(known) == sorted(set(known))  # sorted + de-duplicated
    # Each burst start contributes a 5-day window entirely inside the series.
    for start_i in s.vol_burst_idx:
        for offset in range(5):
            assert start_i + offset in known
    # Every jump index is present.
    for j in s.jump_idx:
        assert j in known
    assert all(0 <= i < 600 for i in known)


def test_injected_anomalies_inflate_return_magnitude() -> None:
    """Injected stress days are materially larger than the calm baseline.

    This is the recoverability guarantee the detectors rely on: at the known
    indices the absolute return is many sigmas above the calm-region std, with
    NO label leaking into the detectors themselves.
    """
    s = generate_injected_series(n_obs=1000, seed=7, base_vol=0.008)
    abs_ret = s.returns.abs().to_numpy()
    known = list(s.known_anomaly_idx())
    calm_mask = np.ones(len(abs_ret), dtype=bool)
    calm_mask[known] = False
    calm_std = float(s.returns.to_numpy()[calm_mask].std())

    # The mean injected-day magnitude clears 3 calm-sigmas comfortably.
    injected_mean = float(abs_ret[known].mean())
    assert injected_mean > 3.0 * calm_std
    # Jumps in particular are ~0.06 in magnitude (vs calm std ~0.008).
    for j in s.jump_idx:
        assert abs(float(s.returns.iloc[j])) > 5.0 * calm_std


def test_injected_series_to_dict_is_json_clean() -> None:
    """``to_dict`` yields a JSON-serializable mapping with ISO dates."""
    import json

    s = generate_injected_series(n_obs=250, seed=2)
    payload = s.to_dict()
    # Round-trips through JSON without custom encoders.
    restored = json.loads(json.dumps(payload))
    assert restored["dates"][0] == s.returns.index[0].isoformat()
    assert len(restored["returns"]) == 250
    assert len(restored["prices"]) == 250
    assert restored["vol_burst_idx"] == list(s.vol_burst_idx)
    assert restored["jump_idx"] == list(s.jump_idx)
    assert restored["known_anomaly_idx"] == list(s.known_anomaly_idx())
    assert restored["meta"]["seed"] == 2


# --------------------------------------------------------------------------- #
# load_prices: offline fallback + validation                                  #
# --------------------------------------------------------------------------- #
def test_load_prices_synthetic_is_offline_and_deterministic() -> None:
    """``source_pref='synthetic'`` returns a deterministic series with NO network."""
    start, end = date(2020, 1, 1), date(2021, 1, 1)
    a, src_a = load_prices("SPY", start, end, source_pref="synthetic", seed=7)
    b, src_b = load_prices("SPY", start, end, source_pref="synthetic", seed=7)
    assert src_a == "synthetic" == src_b
    assert isinstance(a, pd.Series)
    pd.testing.assert_series_equal(a, b)
    assert (a > 0).all()
    assert isinstance(a.index, pd.DatetimeIndex)


def test_load_prices_synthetic_varies_by_ticker() -> None:
    """Different tickers produce different (but reproducible) synthetic paths."""
    start, end = date(2020, 1, 1), date(2021, 1, 1)
    spy, _ = load_prices("SPY", start, end, source_pref="synthetic")
    qqq, _ = load_prices("QQQ", start, end, source_pref="synthetic")
    assert spy.shape == qqq.shape
    assert not np.allclose(spy.to_numpy(), qqq.to_numpy())


def test_load_prices_auto_degrades_to_synthetic_offline() -> None:
    """With no Polygon key/network, ``auto`` falls through to synthetic, no raise."""
    start, end = date(2020, 1, 1), date(2020, 6, 1)
    # use_cache=False forces the live (failing) fetch path, which must degrade.
    series, source = load_prices("SPY", start, end, source_pref="auto", seed=7, use_cache=False)
    assert source == "synthetic"
    assert isinstance(series, pd.Series)
    assert not series.empty


def test_load_prices_polygon_pref_degrades_offline() -> None:
    """``source_pref='polygon'`` degrades to synthetic offline rather than raising."""
    start, end = date(2019, 1, 1), date(2019, 4, 1)
    series, source = load_prices("SPY", start, end, source_pref="polygon", use_cache=False)
    assert source == "synthetic"
    assert not series.empty


@pytest.mark.parametrize(
    ("ticker", "start", "end"),
    [
        ("", date(2020, 1, 1), date(2021, 1, 1)),  # empty ticker
        ("   ", date(2020, 1, 1), date(2021, 1, 1)),  # whitespace ticker
        ("SPY", date(2021, 1, 1), date(2020, 1, 1)),  # end <= start
        ("SPY", date(2020, 1, 1), date(2020, 1, 1)),  # end == start
    ],
)
def test_load_prices_rejects_bad_inputs(ticker: str, start: date, end: date) -> None:
    """Empty tickers and non-increasing date ranges raise ``ValidationError``."""
    with pytest.raises(ValidationError):
        load_prices(ticker, start, end)


def test_default_ticker_is_spy() -> None:
    """The documented default single-ticker universe is SPY."""
    assert DEFAULT_TICKER == "SPY"


# --------------------------------------------------------------------------- #
# load_prices: parquet cache + Polygon-success paths (no network)             #
# --------------------------------------------------------------------------- #
def test_cache_round_trips_close_series(tmp_path: Path) -> None:
    """The private parquet cache helpers round-trip a close series losslessly."""
    from anomaly_detector.data import _read_cache, _write_cache

    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    series = pd.Series(np.linspace(100.0, 110.0, 10), index=idx, name="SPY")
    path = tmp_path / "cache.parquet"

    assert _read_cache(path) is None  # cold cache -> None
    _write_cache(path, series)
    restored = _read_cache(path)
    assert restored is not None
    np.testing.assert_allclose(restored.to_numpy(), series.to_numpy())
    assert isinstance(restored.index, pd.DatetimeIndex)


def test_read_cache_returns_none_on_corrupt_file(tmp_path: Path) -> None:
    """A non-parquet file at the cache path degrades to ``None`` (no raise)."""
    from anomaly_detector.data import _read_cache

    path = tmp_path / "corrupt.parquet"
    path.write_text("not a parquet file")
    assert _read_cache(path) is None


def test_load_prices_uses_polygon_when_fetch_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful (monkeypatched) Polygon fetch is reported as ``polygon``."""
    import anomaly_detector.data as data_mod

    idx = pd.date_range("2020-01-01", periods=8, freq="B")
    fake = pd.Series(np.linspace(300.0, 310.0, 8), index=idx, name="SPY")

    def _fake_fetch(ticker: str, start: date, end: date) -> pd.Series:
        return fake

    monkeypatch.setattr(data_mod, "_fetch_polygon_close", _fake_fetch)
    series, source = load_prices(
        "SPY", date(2020, 1, 1), date(2020, 2, 1), source_pref="polygon", use_cache=False
    )
    assert source == "polygon"
    np.testing.assert_allclose(series.to_numpy(), fake.to_numpy())


def test_load_prices_empty_polygon_result_degrades(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty Polygon series falls through to the synthetic path."""
    import anomaly_detector.data as data_mod

    def _empty_fetch(ticker: str, start: date, end: date) -> pd.Series:
        return pd.Series(dtype="float64", name=ticker)

    monkeypatch.setattr(data_mod, "_fetch_polygon_close", _empty_fetch)
    series, source = load_prices(
        "SPY", date(2020, 1, 1), date(2020, 2, 1), source_pref="auto", use_cache=False
    )
    assert source == "synthetic"
    assert not series.empty


def test_load_prices_warm_cache_short_circuits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A warm parquet cache is returned without calling the live fetch."""
    import anomaly_detector.data as data_mod

    start, end = date(2020, 1, 1), date(2020, 2, 1)
    idx = pd.date_range(start, periods=6, freq="B")
    warm = pd.Series(np.linspace(400.0, 410.0, 6), index=idx, name="SPY")

    cache_file = tmp_path / "warm.parquet"

    def _fixed_path(ticker: str, s: date, e: date) -> Path:
        return cache_file

    monkeypatch.setattr(data_mod, "_cache_path", _fixed_path)
    data_mod._write_cache(cache_file, warm)

    def _boom(ticker: str, s: date, e: date) -> pd.Series:
        raise AssertionError("live fetch must not run on a warm cache hit")

    monkeypatch.setattr(data_mod, "_fetch_polygon_close", _boom)
    series, source = load_prices("SPY", start, end, source_pref="auto", use_cache=True)
    assert source == "polygon"
    np.testing.assert_allclose(series.to_numpy(), warm.to_numpy())


def test_load_prices_writes_cache_on_live_fetch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A live Polygon fetch writes the result to the parquet cache."""
    import anomaly_detector.data as data_mod

    start, end = date(2020, 1, 1), date(2020, 2, 1)
    idx = pd.date_range(start, periods=5, freq="B")
    fetched = pd.Series(np.linspace(500.0, 504.0, 5), index=idx, name="SPY")
    cache_file = tmp_path / "written.parquet"

    def _fixed_path(ticker: str, s: date, e: date) -> Path:
        return cache_file

    def _fetch(ticker: str, s: date, e: date) -> pd.Series:
        return fetched

    monkeypatch.setattr(data_mod, "_cache_path", _fixed_path)
    monkeypatch.setattr(data_mod, "_fetch_polygon_close", _fetch)

    _series, source = load_prices("SPY", start, end, source_pref="polygon", use_cache=True)
    assert source == "polygon"
    assert cache_file.exists()
    written = data_mod._read_cache(cache_file)
    assert written is not None
    np.testing.assert_allclose(written.to_numpy(), fetched.to_numpy())


def test_fetch_polygon_close_raises_on_empty_panel(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_fetch_polygon_close`` raises when the provider returns an empty panel."""
    import anomaly_detector.data as data_mod
    from anomaly_detector.data_providers import polygon as polygon_mod

    class _EmptyProvider:
        def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(polygon_mod, "PolygonProvider", _EmptyProvider)
    with pytest.raises(ValueError, match="no usable price data"):
        data_mod._fetch_polygon_close("SPY", date(2020, 1, 1), date(2020, 2, 1))


def test_fetch_polygon_close_parses_provider_panel(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_fetch_polygon_close`` extracts the ticker column from a provider panel."""
    import anomaly_detector.data as data_mod
    from anomaly_detector.data_providers import polygon as polygon_mod

    idx = pd.date_range("2020-01-01", periods=4, freq="B")
    panel = pd.DataFrame({"SPY": np.linspace(100.0, 103.0, 4)}, index=idx)

    class _PanelProvider:
        def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
            return panel

    monkeypatch.setattr(polygon_mod, "PolygonProvider", _PanelProvider)
    out = data_mod._fetch_polygon_close("SPY", date(2020, 1, 1), date(2020, 2, 1))
    np.testing.assert_allclose(out.to_numpy(), panel["SPY"].to_numpy())
    assert out.name == "SPY"


def test_synthetic_prices_empty_range_is_empty() -> None:
    """A degenerate (weekend-only) range yields an empty synthetic series."""
    from anomaly_detector.data import _synthetic_prices

    # 2021-01-02 and 2021-01-03 are Sat/Sun -> no business days in range.
    out = _synthetic_prices("SPY", date(2021, 1, 2), date(2021, 1, 3), seed=7)
    assert out.empty


def test_cache_path_is_deterministic_and_parquet() -> None:
    """``_cache_path`` is a deterministic, ticker/range-keyed ``.parquet`` path."""
    from anomaly_detector.data import _cache_path

    start, end = date(2020, 1, 1), date(2021, 1, 1)
    p1 = _cache_path("SPY", start, end)
    p2 = _cache_path("SPY", start, end)
    p3 = _cache_path("QQQ", start, end)
    assert p1 == p2  # deterministic
    assert p1 != p3  # keyed by ticker
    assert p1.suffix == ".parquet"
    assert "SPY" in p1.name


def test_read_cache_returns_none_on_empty_frame(tmp_path: Path) -> None:
    """A cached parquet with zero columns degrades to ``None``."""
    from anomaly_detector.data import _read_cache

    path = tmp_path / "empty.parquet"
    pd.DataFrame(index=pd.date_range("2020-01-01", periods=3, freq="B")).to_parquet(path)
    assert _read_cache(path) is None


def test_write_cache_is_noop_on_unwritable_path(tmp_path: Path) -> None:
    """``_write_cache`` swallows write failures (e.g. a file-as-directory parent)."""
    from anomaly_detector.data import _read_cache, _write_cache

    # Make the would-be parent a FILE so ``mkdir``/``to_parquet`` fail; the
    # helper must silently no-op rather than raise.
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory")
    doomed = blocker / "nested" / "out.parquet"
    _write_cache(doomed, pd.Series([1.0, 2.0], name="SPY"))  # must not raise
    assert _read_cache(doomed) is None


def test_compute_returns_rejects_empty_series() -> None:
    """An empty price series raises ``ValidationError``."""
    with pytest.raises(ValidationError, match="non-empty"):
        compute_returns(pd.Series(dtype="float64"))


# --------------------------------------------------------------------------- #
# compute_returns: no-lookahead across input shapes                           #
# --------------------------------------------------------------------------- #
def test_compute_returns_drops_leading_nan_and_matches_pct_change() -> None:
    """Returns equal ``pct_change(fill_method=None)`` with the leading NaN dropped."""
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    prices = pd.Series([100.0, 101.0, 99.0, 103.0, 103.0], index=idx, name="price")
    out = compute_returns(prices)
    expected = prices.pct_change(fill_method=None).iloc[1:]
    assert len(out) == 4
    assert out.isna().sum() == 0
    np.testing.assert_allclose(out.to_numpy(), expected.to_numpy())


def test_compute_returns_does_not_ffill_across_gaps() -> None:
    """A gap (NaN price) must NOT be forward-filled before differencing.

    ffill-then-diff would manufacture a spurious 0.0 return across the gap; with
    ``fill_method=None`` the return spanning the gap is NaN, not a fake zero.
    """
    idx = pd.date_range("2020-01-01", periods=4, freq="B")
    prices = pd.Series([100.0, np.nan, 110.0, 121.0], index=idx, name="price")
    out = compute_returns(prices)
    # Position 1 (NaN price) and position 2 (return spanning the gap) are NaN,
    # never a spurious 0.0 manufactured by forward-filling.
    assert bool(out.isna().any())
    assert not (out == 0.0).any()


def test_compute_returns_accepts_dataframe_and_ndarray() -> None:
    """A single-column DataFrame and a 1-D ndarray are accepted and agree."""
    idx = pd.date_range("2020-01-01", periods=4, freq="B")
    values = [100.0, 102.0, 101.0, 105.0]
    from_series = compute_returns(pd.Series(values, index=idx))
    from_frame = compute_returns(pd.DataFrame({"price": values}, index=idx))
    from_array = compute_returns(np.asarray(values, dtype="float64"))
    np.testing.assert_allclose(from_series.to_numpy(), from_frame.to_numpy())
    np.testing.assert_allclose(from_series.to_numpy(), from_array.to_numpy())


@pytest.mark.parametrize(
    "bad",
    [
        pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}),  # multi-column frame
        np.zeros((3, 2)),  # 2-D ndarray
        "not-prices",  # wrong type
    ],
)
def test_compute_returns_rejects_malformed(bad: object) -> None:
    """Multi-column frames, 2-D arrays, and non-price inputs raise."""
    with pytest.raises(ValidationError):
        compute_returns(bad)  # type: ignore[arg-type]


def test_compute_returns_round_trips_synthetic_prices() -> None:
    """The synthetic injected prices differenced back match the source returns.

    ``prices = base * cumprod(1 + r)`` so ``pct_change`` recovers ``r`` exactly
    (modulo the dropped leading observation) — a no-lookahead consistency check.
    """
    s = generate_injected_series(n_obs=200, seed=5)
    recovered = compute_returns(s.prices)
    np.testing.assert_allclose(
        recovered.to_numpy(), s.returns.iloc[1:].to_numpy(), rtol=1e-9, atol=1e-12
    )


# --------------------------------------------------------------------------- #
# Import purity                                                                #
# --------------------------------------------------------------------------- #
def test_data_module_has_no_module_level_heavy_imports() -> None:
    """The data module imports its lazy/heavy deps ONLY inside functions.

    The Polygon provider (and ``httpx``), the parquet cache (``pyarrow``), and
    ``tempfile``/``pathlib`` for the cache path all live behind function-local
    imports so importing :mod:`anomaly_detector.data` has zero network/heavy
    side effects. We assert this structurally by parsing the module's AST: no
    top-level ``import``/``from`` statement may reference a lazy dependency.

    (Verified by AST rather than ``sys.modules`` because ``pyarrow`` is pulled in
    eagerly by ``import pandas`` itself, which would mask a genuine regression.)
    """
    import ast
    import pathlib

    import anomaly_detector.data as data_mod

    source = pathlib.Path(data_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    lazy = {"httpx", "pyarrow", "sklearn", "plotly", "typer", "tempfile"}

    module_level_imports: set[str] = set()
    for node in tree.body:  # only top-level statements
        if isinstance(node, ast.Import):
            module_level_imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            module_level_imports.add(node.module.split(".")[0])

    offenders = module_level_imports & lazy
    assert not offenders, f"lazy deps imported at module level: {sorted(offenders)}"
    # The Polygon provider must likewise be imported lazily, never at top level.
    assert "data_providers" not in {
        node.module.split(".")[1]
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("anomaly_detector.")
        and "." in node.module
    }
