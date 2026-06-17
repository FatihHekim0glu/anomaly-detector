"""Regression tests: golden anomaly recovery + honest-headline guards.

Pins behaviour on the fixed synthetic injected-anomaly series:

- golden recovery — both detectors recover the KNOWN injected stress indices
  (vol bursts / jumps) at better-than-chance rate, with NO lookahead;
- honest-headline guard — day-level agreement is MODEST (Jaccard within a
  documented band, ~0.3-0.5) and proxy precision is LOW, so the summary can
  never imply a tradable signal.

Skipped until the pipeline is implemented (the scaffold ships green). The
parallel author removes the skip and fills the body against the
``injected_anomalies`` fixture.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.regression


@pytest.mark.skip(reason="scaffold: implement once detectors + agreement are real")
def test_detectors_recover_known_injected_indices() -> None:
    """Both detectors recover the injected stress indices better than chance."""
    raise NotImplementedError


@pytest.mark.skip(reason="scaffold: implement once compute_agreement is real")
def test_honest_headline_modest_jaccard_low_precision() -> None:
    """Jaccard stays modest and proxy precision stays low (the honest null)."""
    raise NotImplementedError
