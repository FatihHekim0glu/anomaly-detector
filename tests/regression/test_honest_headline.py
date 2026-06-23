"""Regression guard: the honest, DESCRIPTIVE headline must hold.

The whole point of this tool is an honest-null one: the two independent detectors
agree on a *core* of stress days, and the load-bearing honest claim is that their
precision against a transparent ``|z-return| > 3`` proxy label is LOW - flags are
diagnostic, not tradable. These guards pin those claims on the SHIPPED causal
walk-forward path (:func:`anomaly_detector.scan.run_anomaly_scan`) over the exact
canonical series the README headline uses, so the tool a user runs, the public
API, the FastAPI router, and the documented numbers can never silently diverge.

Honest reconciliation: the walk-forward Jaccard (~0.73) is HIGHER than the
retired one-shot 60/40 simple-split number (~0.50) the console script used to
report. The walk-forward refit is the documented path, so its measured agreement
is what we pin - reported openly, not hidden. The detectors still only agree on a
core, not a signal: proxy precision stays LOW, which is the claim that matters.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression

# Canonical README series: generate_injected_series(n_obs=1200, seed=7). The
# measured causal WALK-FORWARD agreement on it (run_anomaly_scan, default
# n_folds=4, window=21, contamination=0.02, seed=7) is Jaccard ~0.7308. A value
# INSIDE this tight band is the honest headline; near 1.0 would imply the
# detectors are redundant, near 0.0 that they are unrelated - both would undercut
# the "agree on a core" story. The band is centred on the measured 0.7308 with a
# small tolerance so an unintended drift in the shipped path trips the guard.
_JACCARD_LOW = 0.68
_JACCARD_HIGH = 0.78

# The transparent |z-return| > 3 proxy is NOT a ground-truth label, so precision
# against it must stay LOW - flags are diagnostic, not a clean predictor. The
# measured walk-forward precision is ~0.0345; the ceiling keeps the LOAD-BEARING
# honest claim pinned with comfortable headroom.
_PROXY_PRECISION_CEILING = 0.10

# Canonical README series parameters (single source of truth for this guard).
_CANONICAL_N_OBS = 1200
_CANONICAL_SEED = 7


@pytest.mark.parametrize("contamination", [0.02])
def test_honest_headline_modest_jaccard_low_precision(
    default_window: int,
    contamination: float,
) -> None:
    """Walk-forward Jaccard stays in-band and proxy precision stays LOW.

    Runs the SHIPPED public ``run_anomaly_scan`` walk-forward path on the exact
    canonical README series, so this guard pins what the tool actually reports.
    """
    from anomaly_detector.data import generate_injected_series
    from anomaly_detector.scan import run_anomaly_scan

    inj = generate_injected_series(n_obs=_CANONICAL_N_OBS, seed=_CANONICAL_SEED)

    scan = run_anomaly_scan(
        prices=inj.prices,
        detector="both",
        contamination=contamination,
        window=default_window,
        seed=_CANONICAL_SEED,
    )
    summary = scan.agreement

    # (1) Both detectors fire on a non-trivial number of OOS days (the run is
    #     meaningful, not a degenerate "flag nothing" pass).
    assert summary.n_flags_a > 0
    assert summary.n_flags_b > 0

    # (2) Day-level walk-forward agreement is inside the documented band, neither
    #     redundant (~1.0) nor unrelated (~0.0).
    assert _JACCARD_LOW <= summary.jaccard <= _JACCARD_HIGH, (
        f"walk-forward Jaccard {summary.jaccard:.4f} escaped the honest band "
        f"[{_JACCARD_LOW}, {_JACCARD_HIGH}] - the headline would mislead."
    )

    # (3) The LOAD-BEARING claim: precision against the transparent proxy label is
    #     LOW - flags are diagnostic, not a clean tradable predictor.
    assert summary.proxy_precision <= _PROXY_PRECISION_CEILING, (
        f"proxy precision {summary.proxy_precision:.4f} is too high - the "
        "summary would imply the flags cleanly predict the proxy."
    )

    # (4) The summary stays inside its honest, JSON-safe envelope.
    assert 0.0 <= summary.jaccard <= 1.0
    assert 0.0 <= summary.proxy_precision <= 1.0
    assert 0.0 <= summary.proxy_recall <= 1.0
    assert 0.0 <= summary.regime_alignment <= 1.0
