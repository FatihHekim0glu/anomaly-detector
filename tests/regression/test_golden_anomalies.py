"""Regression tests: golden anomaly recovery + honest-headline guards.

Pins behaviour on the fixed synthetic injected-anomaly series:

- golden recovery — both detectors recover the KNOWN injected stress indices
  (vol bursts / jumps) at a better-than-chance rate, with NO lookahead;
- honest-headline guard — day-level agreement is MODEST and proxy precision is
  LOW, so the summary can never imply a tradable signal.

The detectors are fitted on the CALM front of the series (which contains no
injected anomalies) and scored on the disjoint back slice (which does), so the
run is strictly no-lookahead: the train threshold never sees an injected day.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression

#: Split bar: every injected index is >= 520, so rows before this are the calm
#: front (the no-lookahead train slice) and the threshold never sees a stress day.
_SPLIT_BAR = 500

#: Documented modest-Jaccard band (~0.3-0.5 at the default 2% contamination);
#: a wider tolerance keeps the guard robust without letting it drift to ~1.0/~0.0.
_JACCARD_LOW = 0.20
_JACCARD_HIGH = 0.65

#: The transparent |z-return| > 3 proxy is NOT ground truth, so precision against
#: it must stay LOW — flags are diagnostic, not a clean predictor.
_PROXY_PRECISION_CEILING = 0.20


def _calm_front_split(prices, features, returns):
    """Causal split: calm-front TRAIN (no injected days) and stress-back OOS."""
    split_label = prices.index[_SPLIT_BAR]
    train = features.loc[features.index < split_label]
    test = features.loc[features.index >= split_label]
    returns_oos = returns.reindex(test.index)
    return train, test, returns_oos


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
            f"false-positive rate ({fp_non_injected:.3f}) — no recovery."
        )


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
    train, test, returns_oos = _calm_front_split(prices, features, returns)

    det_if = IsolationForestDetector(contamination=contamination, seed=7).fit(train)
    det_ae = PCAAutoencoderDetector(contamination=contamination, seed=7).fit(train)
    summary = compute_agreement(
        det_if.score(test), det_ae.score(test), returns_oos, window=default_window
    )

    assert summary.n_flags_a > 0
    assert summary.n_flags_b > 0
    assert _JACCARD_LOW <= summary.jaccard <= _JACCARD_HIGH, (
        f"Jaccard {summary.jaccard:.3f} escaped the honest modest band."
    )
    assert summary.proxy_precision <= _PROXY_PRECISION_CEILING, (
        f"proxy precision {summary.proxy_precision:.3f} is too high."
    )
    assert 0.0 <= summary.proxy_recall <= 1.0
    assert 0.0 <= summary.regime_alignment <= 1.0
