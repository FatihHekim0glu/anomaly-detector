"""Unit tests for the descriptive detector-agreement diagnostics.

These exercise :mod:`anomaly_detector.evaluation.agreement` directly, building
flag / score series by hand (and small synthetic :class:`AnomalyResult` objects)
so the agreement math is verified WITHOUT depending on the real detectors. The
checks cover Jaccard correctness, proxy precision/recall against the transparent
``|z-return| > 3`` label, regime alignment with the known stress windows, and the
assembled :func:`compute_agreement` summary plus its ``to_dict`` serialization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from anomaly_detector.detectors.result import AnomalyResult
from anomaly_detector.evaluation.agreement import (
    KNOWN_STRESS_WINDOWS,
    PROXY_Z_THRESHOLD,
    AgreementResult,
    compute_agreement,
    jaccard_index,
    proxy_precision_recall,
    regime_alignment,
)

pytestmark = pytest.mark.unit


def _flags(index: pd.DatetimeIndex, true_positions: list[int]) -> pd.Series:
    """Build a boolean flag Series with ``True`` at the given positions."""
    arr = np.zeros(len(index), dtype=bool)
    for p in true_positions:
        arr[p] = True
    return pd.Series(arr, index=index, name="anomaly_flag")


def _result(
    scores: pd.Series,
    flags: pd.Series,
    *,
    detector: str,
    threshold: float = 0.0,
) -> AnomalyResult:
    """Wrap score/flag series into an :class:`AnomalyResult` for agreement tests."""
    return AnomalyResult(
        scores=scores,
        flags=flags,
        threshold=threshold,
        detector=detector,
        contamination=0.02,
        n_train=100,
        n_test=int(scores.shape[0]),
    )


# --------------------------------------------------------------------------- #
# jaccard_index                                                               #
# --------------------------------------------------------------------------- #
def test_jaccard_identical_flag_sets_is_one() -> None:
    """Two identical (non-empty) flag sets have Jaccard exactly 1.0."""
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    a = _flags(idx, [1, 3, 5])
    assert jaccard_index(a, a.copy()) == 1.0


def test_jaccard_disjoint_flag_sets_is_zero() -> None:
    """Disjoint flag sets (non-empty union) have Jaccard 0.0."""
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    a = _flags(idx, [1, 2])
    b = _flags(idx, [7, 8])
    assert jaccard_index(a, b) == 0.0


def test_jaccard_partial_overlap_is_correct() -> None:
    """|A and B| / |A or B| matches a hand-computed value."""
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    a = _flags(idx, [1, 2, 3])  # {1,2,3}
    b = _flags(idx, [2, 3, 4])  # {2,3,4}; intersection {2,3}=2, union 4
    assert jaccard_index(a, b) == pytest.approx(2.0 / 4.0)


def test_jaccard_empty_union_defined_as_zero() -> None:
    """When neither detector flags anything, Jaccard is 0.0 (not NaN)."""
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    empty = _flags(idx, [])
    assert jaccard_index(empty, empty.copy()) == 0.0


def test_jaccard_ignores_nan_flags() -> None:
    """NaN flag entries are treated as not-flagged, not as truthy."""
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    a = pd.Series([True, np.nan, False, True, np.nan], index=idx)
    b = pd.Series([True, True, False, False, np.nan], index=idx)
    # A = {0, 3}, B = {0, 1}; intersection {0}=1, union {0,1,3}=3.
    assert jaccard_index(a, b) == pytest.approx(1.0 / 3.0)


# --------------------------------------------------------------------------- #
# proxy_precision_recall                                                      #
# --------------------------------------------------------------------------- #
def test_proxy_precision_recall_on_injected_anomalies(
    injected_anomalies: object,
    default_window: int,
) -> None:
    """Flags equal to the proxy label itself score perfect precision/recall."""
    returns = injected_anomalies.returns  # type: ignore[attr-defined]
    from anomaly_detector.features.engineer import causal_return_zscore

    zscore = causal_return_zscore(returns, window=default_window)
    proxy = (zscore.abs() > PROXY_Z_THRESHOLD).fillna(False)
    # Use the proxy itself as the flags: precision and recall must both be 1.0.
    precision, recall = proxy_precision_recall(proxy, returns, window=default_window)
    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(1.0)
    # The proxy must actually fire on the injected series (a meaningful test).
    assert int(proxy.sum()) > 0


def test_proxy_precision_low_when_flags_miss_proxy(
    injected_anomalies: object,
    default_window: int,
) -> None:
    """Flagging calm days the proxy ignores yields low precision (honest null)."""
    returns = injected_anomalies.returns  # type: ignore[attr-defined]
    from anomaly_detector.features.engineer import causal_return_zscore

    zscore = causal_return_zscore(returns, window=default_window)
    # Flag the FIRST 30 defined (post-warm-up) calm days, almost none of which
    # are proxy-positive: precision should be far below 1.
    defined = zscore.dropna().index
    flagged = pd.Series(False, index=returns.index, name="anomaly_flag")
    flagged.loc[defined[:30]] = True
    precision, _ = proxy_precision_recall(flagged, returns, window=default_window)
    assert 0.0 <= precision < 0.5


def test_proxy_precision_recall_zero_when_no_flags(
    clean_series: pd.Series,
    default_window: int,
) -> None:
    """No flags -> precision and recall are both 0.0 (empty numerator)."""
    flags = pd.Series(False, index=clean_series.index, name="anomaly_flag")
    precision, recall = proxy_precision_recall(flags, clean_series, window=default_window)
    assert precision == 0.0
    assert recall == 0.0


def test_proxy_precision_recall_empty_overlap_is_zero(default_window: int) -> None:
    """Disjoint flag/return indexes -> (0.0, 0.0) rather than an error."""
    flag_idx = pd.date_range("2020-01-01", periods=50, freq="B")
    ret_idx = pd.date_range("2010-01-01", periods=50, freq="B")
    flags = _flags(flag_idx, [10, 20])
    returns = pd.Series(np.random.default_rng(0).standard_normal(50) * 0.01, index=ret_idx)
    precision, recall = proxy_precision_recall(flags, returns, window=default_window)
    assert (precision, recall) == (0.0, 0.0)


# --------------------------------------------------------------------------- #
# regime_alignment                                                            #
# --------------------------------------------------------------------------- #
def test_regime_alignment_all_inside_window_is_one() -> None:
    """All flagged days inside a known stress window -> alignment 1.0."""
    # COVID window is 2020-02-20 .. 2020-04-30.
    idx = pd.date_range("2020-03-02", periods=10, freq="B")
    flags = _flags(idx, list(range(10)))
    assert regime_alignment(flags) == pytest.approx(1.0)


def test_regime_alignment_all_outside_window_is_zero() -> None:
    """Flagged days far from any stress window -> alignment 0.0."""
    idx = pd.date_range("2017-01-02", periods=10, freq="B")  # calm 2017
    flags = _flags(idx, list(range(10)))
    assert regime_alignment(flags) == 0.0


def test_regime_alignment_partial() -> None:
    """A mix of in-window / out-of-window flagged days gives the right fraction."""
    in_window = pd.date_range("2020-03-02", periods=3, freq="B")  # inside COVID
    out_window = pd.date_range("2017-06-01", periods=1, freq="B")  # calm
    idx = in_window.append(out_window)
    flags = _flags(idx, [0, 1, 2, 3])  # 3 inside, 1 outside -> 3/4
    assert regime_alignment(flags) == pytest.approx(3.0 / 4.0)


def test_regime_alignment_no_flags_is_zero() -> None:
    """No flagged days -> alignment 0.0 (empty numerator and denominator)."""
    idx = pd.date_range("2020-03-02", periods=5, freq="B")
    flags = _flags(idx, [])
    assert regime_alignment(flags) == 0.0


def test_known_stress_windows_are_well_formed() -> None:
    """Each known stress window is a valid (lo <= hi) ISO date range."""
    for lo, hi in KNOWN_STRESS_WINDOWS:
        assert pd.Timestamp(lo) <= pd.Timestamp(hi)


# --------------------------------------------------------------------------- #
# compute_agreement + to_dict                                                 #
# --------------------------------------------------------------------------- #
def test_compute_agreement_assembles_summary(default_window: int) -> None:
    """compute_agreement wires Jaccard / proxy / regime into one summary."""
    idx = pd.date_range("2020-01-01", periods=60, freq="B")
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.standard_normal(60) * 0.01, index=idx, name="return")

    scores_a = pd.Series(rng.standard_normal(60), index=idx, name="anomaly_score")
    scores_b = pd.Series(rng.standard_normal(60), index=idx, name="anomaly_score")
    flags_a = _flags(idx, [10, 20, 30])  # {10,20,30}
    flags_b = _flags(idx, [20, 30, 40])  # {20,30,40}
    res_a = _result(scores_a, flags_a, detector="iforest")
    res_b = _result(scores_b, flags_b, detector="autoencoder")

    summary = compute_agreement(res_a, res_b, returns, window=default_window)
    assert isinstance(summary, AgreementResult)
    # Jaccard of {10,20,30} vs {20,30,40}: intersection=2, union=4.
    assert summary.jaccard == pytest.approx(0.5)
    assert summary.overlap_count == 2
    assert summary.n_flags_a == 3
    assert summary.n_flags_b == 3
    assert 0.0 <= summary.proxy_precision <= 1.0
    assert 0.0 <= summary.proxy_recall <= 1.0
    assert 0.0 <= summary.regime_alignment <= 1.0
    assert summary.meta["detector_a"] == "iforest"
    assert summary.meta["detector_b"] == "autoencoder"


def test_compute_agreement_top_dates_prefer_agreement(default_window: int) -> None:
    """Top-anomaly dates list days flagged by BOTH detectors ahead of singles."""
    idx = pd.date_range("2020-01-01", periods=40, freq="B")
    returns = pd.Series(0.0, index=idx, name="return")

    # Build scores so positions 5 and 6 are the most anomalous on both detectors.
    base = np.linspace(0.0, 1.0, 40)
    scores_a = pd.Series(base.copy(), index=idx)
    scores_b = pd.Series(base.copy(), index=idx)
    # Both detectors flag {5, 6} (agreed) and each flags one extra single day.
    flags_a = _flags(idx, [5, 6, 38])
    flags_b = _flags(idx, [5, 6, 39])
    res_a = _result(scores_a, flags_a, detector="iforest")
    res_b = _result(scores_b, flags_b, detector="autoencoder")

    summary = compute_agreement(res_a, res_b, returns, window=default_window)
    agreed_iso = {pd.Timestamp(idx[5]).date().isoformat(), pd.Timestamp(idx[6]).date().isoformat()}
    # The two agreed dates appear before any single-detector date in the list.
    first_two = set(summary.top_anomaly_dates[:2])
    assert first_two == agreed_iso


def test_compute_agreement_to_dict_is_json_safe(default_window: int) -> None:
    """to_dict returns only JSON-serializable, finite-or-None scalar types."""
    idx = pd.date_range("2020-01-01", periods=30, freq="B")
    returns = pd.Series(0.0, index=idx, name="return")
    scores = pd.Series(np.arange(30, dtype="float64"), index=idx)
    flags = _flags(idx, [5, 15, 25])
    res_a = _result(scores, flags, detector="iforest")
    res_b = _result(scores.copy(), _flags(idx, [15, 25]), detector="autoencoder")

    payload = compute_agreement(res_a, res_b, returns, window=default_window).to_dict()
    import json

    encoded = json.dumps(payload)  # must not raise
    assert isinstance(encoded, str)
    assert payload["overlap_count"] == 2
    assert isinstance(payload["jaccard"], float)
    assert all(isinstance(d, str) for d in payload["top_anomaly_dates"])


def test_compute_agreement_disjoint_score_index_yields_no_top_dates(
    default_window: int,
) -> None:
    """Detectors scored on disjoint indexes -> empty top-dates, still a summary."""
    idx_a = pd.date_range("2020-01-01", periods=30, freq="B")
    idx_b = pd.date_range("2030-01-01", periods=30, freq="B")  # no overlap
    returns = pd.Series(0.0, index=idx_a, name="return")
    res_a = _result(
        pd.Series(np.arange(30, dtype="float64"), index=idx_a),
        _flags(idx_a, [5, 10]),
        detector="iforest",
    )
    res_b = _result(
        pd.Series(np.arange(30, dtype="float64"), index=idx_b),
        _flags(idx_b, [5, 10]),
        detector="autoencoder",
    )
    summary = compute_agreement(res_a, res_b, returns, window=default_window)
    assert summary.top_anomaly_dates == []
    # Flag counts are still reported even with no shared scoring dates.
    assert summary.n_flags_a == 2
    assert summary.n_flags_b == 2


def test_compute_agreement_pure_noise_low_agreement(
    pure_noise: pd.Series,
    default_window: int,
) -> None:
    """On pure noise with disjoint random flags, agreement is near chance/low."""
    idx = pure_noise.index
    rng = np.random.default_rng(11)
    scores_a = pd.Series(rng.standard_normal(len(idx)), index=idx)
    scores_b = pd.Series(rng.standard_normal(len(idx)), index=idx)
    # Two independent random 2% flag sets over the null series.
    pos_a = sorted(rng.choice(len(idx), size=20, replace=False).tolist())
    pos_b = sorted(rng.choice(len(idx), size=20, replace=False).tolist())
    res_a = _result(scores_a, _flags(idx, pos_a), detector="iforest")
    res_b = _result(scores_b, _flags(idx, pos_b), detector="autoencoder")

    summary = compute_agreement(res_a, res_b, pure_noise, window=default_window)
    # Independent 2% flag sets over 1000 days overlap on essentially nothing.
    assert summary.jaccard < 0.2
