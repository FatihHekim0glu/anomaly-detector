"""Regression tests: golden anomaly recovery + honest-headline guards.

Pins behaviour on the fixed synthetic injected-anomaly series:

- golden recovery - both detectors recover the KNOWN injected stress indices
  (vol bursts / jumps) at a better-than-chance rate, with NO lookahead;
- honest-headline guard - on the SHIPPED causal walk-forward path the day-level
  agreement sits in a documented band and proxy precision stays LOW, so the
  summary can never imply a tradable signal.

The golden-recovery test fits on the CALM front of the series (which contains no
injected anomalies) and scores the disjoint back slice (which does), so it is
strictly no-lookahead. The honest-headline guard runs the public walk-forward
entrypoint (:func:`anomaly_detector.scan.run_anomaly_scan`) directly, so it pins
exactly what the tool, the public API, and the README report.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression

#: Canonical README series: generate_injected_series(n_obs=1200, seed=7). The
#: measured causal WALK-FORWARD agreement on it (run_anomaly_scan defaults) is
#: Jaccard ~0.7308; a tight band trips on any unintended drift. This is HIGHER
#: than the retired one-shot simple-split number (~0.50) - honest reconciliation:
#: the walk-forward refit is the documented path, so its agreement is what we pin.
_CANONICAL_N_OBS = 1200
_CANONICAL_SEED = 7
_JACCARD_LOW = 0.68
_JACCARD_HIGH = 0.78

#: The transparent |z-return| > 3 proxy is NOT ground truth, so precision against
#: it must stay LOW - the LOAD-BEARING honest claim (flags are diagnostic, not a
#: clean predictor). Measured walk-forward precision is ~0.0345.
_PROXY_PRECISION_CEILING = 0.10


#: Causal tolerance (in days) around each injected index. A feature row at day
#: ``t`` reflects data through ``t - 1`` (the ``.shift(1)`` chokepoint) and a
#: vol-burst spans several bars, so an injected anomaly at index ``i`` surfaces
#: in the flag at ``i`` .. ``i + 2``; this window absorbs the lag without leaking.
_RECOVERY_LO: int = -1
_RECOVERY_HI: int = 2


def test_detectors_recover_known_injected_indices() -> None:
    """Both detectors recover the injected stress indices better than chance.

    Runs on the FIXED golden synthetic series from
    :func:`anomaly_detector.data.generate_injected_series`, whose anomalies live
    in the back half at recorded indices. The detectors are fitted on the calm
    front (their threshold never sees an injected day) and score the disjoint
    back slice. Recovery is RECALL on the injected days (with a small causal
    tolerance) versus the FALSE-POSITIVE rate on the non-injected OOS days: a
    real detector flags the injected core at a clearly higher rate than the calm
    background. No label is ever fed to the detectors.
    """
    from anomaly_detector.data import generate_injected_series
    from anomaly_detector.detectors.autoencoder import PCAAutoencoderDetector
    from anomaly_detector.detectors.iforest import IsolationForestDetector
    from anomaly_detector.features.engineer import engineer_features

    injected = generate_injected_series(n_obs=1200, seed=7, jump_size=0.08, burst_vol_mult=6.0)
    prices = injected.prices
    features = engineer_features(prices, window=21)

    # Anchored calm-front split: the injections start past the midpoint, so a
    # 55% cut keeps the TRAIN slice entirely calm (no injected day in fit).
    cut = int(len(features) * 0.55)
    train = features.iloc[:cut]
    test = features.iloc[cut:]

    n_obs = len(prices.index)
    known_window: set[int] = set()
    for i in injected.known_anomaly_idx():
        for offset in range(_RECOVERY_LO, _RECOVERY_HI + 1):
            pos = i + offset
            if 0 <= pos < n_obs:
                known_window.add(pos)

    test_positions = [prices.index.get_loc(d) for d in test.index]
    injected_mask = [pos in known_window for pos in test_positions]
    n_injected = sum(injected_mask)
    assert n_injected > 0, "fixture sanity: injected days must land in the OOS slice"

    for detector_cls in (IsolationForestDetector, PCAAutoencoderDetector):
        det = detector_cls(contamination=0.05, seed=7).fit(train)
        flags = det.score(test).flags.fillna(False).astype(bool).to_numpy()
        assert flags.any(), f"{detector_cls.__name__} flagged nothing"

        recall_injected = (
            sum(flag for flag, is_inj in zip(flags, injected_mask, strict=True) if is_inj)
            / n_injected
        )
        fp_non_injected = sum(
            flag for flag, is_inj in zip(flags, injected_mask, strict=True) if not is_inj
        ) / (len(flags) - n_injected)

        assert recall_injected > fp_non_injected, (
            f"{detector_cls.__name__} recall on injected days "
            f"({recall_injected:.3f}) is no better than the calm-background "
            f"false-positive rate ({fp_non_injected:.3f}) - no recovery."
        )


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

    assert summary.n_flags_a > 0
    assert summary.n_flags_b > 0
    assert _JACCARD_LOW <= summary.jaccard <= _JACCARD_HIGH, (
        f"walk-forward Jaccard {summary.jaccard:.4f} escaped the honest band."
    )
    assert summary.proxy_precision <= _PROXY_PRECISION_CEILING, (
        f"proxy precision {summary.proxy_precision:.4f} is too high."
    )
    assert 0.0 <= summary.proxy_recall <= 1.0
    assert 0.0 <= summary.regime_alignment <= 1.0
