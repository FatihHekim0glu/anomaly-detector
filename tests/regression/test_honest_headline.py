"""Regression guard: the honest, DESCRIPTIVE headline must hold.

The whole point of this tool is an honest-null one: the two independent detectors
agree on a *core* of stress days, but their day-level agreement is only MODEST and
their precision against a transparent ``|z-return| > 3`` proxy label is LOW. These
guards pin those claims on the seeded ``injected_anomalies`` fixture so the summary
can never silently drift into implying a tradable signal.

The detectors are fitted on the CALM front of the series (which contains no
injected anomalies) and scored on the disjoint back slice (which does), so the
run is strictly no-lookahead: the train threshold never sees an injected day.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression

# Documented modest-Jaccard band from the build brief (~0.3-0.5 at the default
# 2% contamination). A value INSIDE this band is the honest headline; a value
# near 1.0 would imply the detectors are redundant, near 0.0 that they are
# unrelated — both would undercut the "agree on a modest core" story.
_JACCARD_LOW = 0.20
_JACCARD_HIGH = 0.65

# The transparent |z-return| > 3 proxy is NOT a ground-truth label, so precision
# against it must stay LOW — flags are diagnostic, not a clean predictor.
_PROXY_PRECISION_CEILING = 0.20


def _train_test_split(features, returns, *, split_label):
    """Causal split: TRAIN on dates strictly before ``split_label``, OOS after."""
    train = features.loc[features.index < split_label]
    test = features.loc[features.index >= split_label]
    returns_oos = returns.reindex(test.index)
    return train, test, returns_oos


@pytest.mark.parametrize("contamination", [0.02])
def test_honest_headline_modest_jaccard_low_precision(
    injected_anomalies: object,
    default_window: int,
    contamination: float,
) -> None:
    """Jaccard stays modest and proxy precision stays low (the honest null)."""
    from anomaly_detector.detectors.autoencoder import PCAAutoencoderDetector
    from anomaly_detector.detectors.iforest import IsolationForestDetector
    from anomaly_detector.evaluation.agreement import compute_agreement
    from anomaly_detector.features.engineer import engineer_features

    prices = injected_anomalies.prices  # type: ignore[attr-defined]
    returns = injected_anomalies.returns  # type: ignore[attr-defined]

    features = engineer_features(prices, window=default_window)
    # Split at the 500th bar: every injected index is >= 520, so the TRAIN slice
    # is the calm front and the threshold is never fitted on an injected day.
    split_label = prices.index[500]
    train, test, returns_oos = _train_test_split(features, returns, split_label=split_label)

    det_if = IsolationForestDetector(contamination=contamination, seed=7).fit(train)
    det_ae = PCAAutoencoderDetector(contamination=contamination, seed=7).fit(train)
    res_if = det_if.score(test)
    res_ae = det_ae.score(test)

    summary = compute_agreement(res_if, res_ae, returns_oos, window=default_window)

    # (1) Both detectors fire on a non-trivial number of OOS days (the run is
    #     meaningful, not a degenerate "flag nothing" pass).
    assert summary.n_flags_a > 0
    assert summary.n_flags_b > 0

    # (2) Day-level agreement is MODEST — inside the documented band, neither
    #     redundant (~1.0) nor unrelated (~0.0).
    assert _JACCARD_LOW <= summary.jaccard <= _JACCARD_HIGH, (
        f"Jaccard {summary.jaccard:.3f} escaped the honest modest band "
        f"[{_JACCARD_LOW}, {_JACCARD_HIGH}] — the headline would mislead."
    )

    # (3) Precision against the transparent proxy label is LOW — flags are
    #     diagnostic, not a clean tradable predictor.
    assert summary.proxy_precision <= _PROXY_PRECISION_CEILING, (
        f"proxy precision {summary.proxy_precision:.3f} is too high — the "
        "summary would imply the flags cleanly predict the proxy."
    )

    # (4) The summary stays inside its honest, JSON-safe envelope.
    assert 0.0 <= summary.jaccard <= 1.0
    assert 0.0 <= summary.proxy_precision <= 1.0
    assert 0.0 <= summary.proxy_recall <= 1.0
    assert 0.0 <= summary.regime_alignment <= 1.0
