"""Property tests (Hypothesis): the four leakage/correctness invariants.

The brief's four required invariants for the causal feature/detector pipeline:

(a) future-perturbation invariance — mutating bars strictly after day ``t`` never
    changes the score/flag AT ``t`` (the core no-lookahead guarantee);
(b) prefix-determinism — scoring a prefix yields the same per-day values as
    scoring the full series, restricted to that prefix;
(c) scale-invariance of the z-features — multiplying the whole price path by a
    positive constant leaves the standardized features unchanged;
(d) monotonicity of flag count in ``contamination`` — a higher contamination
    never yields fewer flags.

Skipped until the feature engineer and detectors are implemented (the scaffold
ships green). The parallel author removes the skip and fills each body with a
Hypothesis ``@given`` strategy over seeded synthetic series.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.property


@pytest.mark.skip(reason="scaffold: implement once engineer_features/detectors are real")
def test_future_perturbation_invariance() -> None:
    """Mutating bars after ``t`` must not change the score/flag at ``t``."""
    raise NotImplementedError


@pytest.mark.skip(reason="scaffold: implement once engineer_features/detectors are real")
def test_prefix_determinism() -> None:
    """A prefix scores identically to the full series restricted to the prefix."""
    raise NotImplementedError


@pytest.mark.skip(reason="scaffold: implement once engineer_features is real")
def test_zscore_scale_invariance() -> None:
    """Scaling the price path by a positive constant leaves z-features unchanged."""
    raise NotImplementedError


@pytest.mark.skip(reason="scaffold: implement once detectors are real")
def test_flag_count_monotonic_in_contamination() -> None:
    """Higher ``contamination`` never yields fewer flags."""
    raise NotImplementedError
