"""Integration test: full causal walk-forward run on the synthetic fixture.

Exercises the end-to-end pipeline with NO network: synthesize an injected
series -> engineer causal features -> causal train/OOS split -> fit each detector
on TRAIN, score the disjoint OOS slice -> assemble the descriptive agreement
summary -> serialize every result via ``to_dict`` (JSON-clean).

Skipped until the pipeline is implemented (the scaffold ships green). The
parallel author removes the skip and fills the body against the
``injected_anomalies`` fixture.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skip(reason="scaffold: implement once the full pipeline is real")
def test_end_to_end_walk_forward_on_synthetic_fixture() -> None:
    """A full causal walk-forward run produces JSON-serializable results."""
    raise NotImplementedError
