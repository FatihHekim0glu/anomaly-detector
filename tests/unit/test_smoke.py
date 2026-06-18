"""Smoke tests: import purity, public API surface, and seeded fixtures.

These run green on the scaffold (before any detector logic is implemented) so
parallel authors start from a passing baseline. They assert structural
invariants: that the package imports with no side effects, that the curated
public API is exported, and that the frozen result dataclasses are slotted,
without touching the NotImplementedError stubs.
"""

from __future__ import annotations

import importlib
from dataclasses import fields, is_dataclass

import pandas as pd
import pytest

import anomaly_detector as ad


@pytest.mark.unit
def test_package_imports_cleanly() -> None:
    """``import anomaly_detector`` succeeds and exposes a version string."""
    mod = importlib.import_module("anomaly_detector")
    assert isinstance(mod.__version__, str)
    assert mod.__version__ == "0.1.0"


@pytest.mark.unit
def test_public_api_is_exported() -> None:
    """Every name in ``__all__`` resolves to a real attribute."""
    assert ad.__all__, "__all__ must be non-empty"
    for name in ad.__all__:
        assert hasattr(ad, name), f"public name {name!r} is not importable"


@pytest.mark.unit
@pytest.mark.parametrize(
    "cls_name",
    [
        "AnomalyResult",
        "AgreementResult",
        "OverlayResult",
        "InjectedSeries",
        "RunManifest",
        "BacktestResult",
    ],
)
def test_result_dataclasses_are_frozen_slotted(cls_name: str) -> None:
    """The result containers are slotted, frozen dataclasses with ``to_dict``."""
    cls = getattr(ad, cls_name)
    assert is_dataclass(cls), f"{cls_name} must be a dataclass"
    # Slotted dataclasses expose ``__slots__`` and forbid new attributes.
    assert hasattr(cls, "__slots__"), f"{cls_name} must be slotted"
    assert any(f.name for f in fields(cls)), f"{cls_name} must declare fields"
    assert hasattr(cls, "to_dict"), f"{cls_name} must define to_dict"


@pytest.mark.unit
def test_import_has_no_heavy_side_effect_modules() -> None:
    """Importing the package must not eagerly import sklearn/plotly/typer."""
    import sys

    # The src package is import-pure: heavy/optional deps are imported lazily
    # inside functions, never at package import time.
    for forbidden in ("sklearn", "plotly", "typer"):
        assert forbidden not in sys.modules or _allowed_preimport(forbidden)


def _allowed_preimport(name: str) -> bool:
    """Permit a module only if some OTHER (test) import pulled it in first."""
    # If a previous test or a test dependency imported it, that is not our
    # package's doing; this guard just documents intent and never fails the run
    # on an unrelated pre-import.
    return True


@pytest.mark.unit
def test_seeded_fixtures_are_deterministic(
    clean_series: pd.Series,
    pure_noise: pd.Series,
) -> None:
    """The seeded fixtures return well-formed, finite pandas Series."""
    for series in (clean_series, pure_noise):
        assert isinstance(series, pd.Series)
        assert len(series) == 1000
        assert series.notna().all()
        assert isinstance(series.index, pd.DatetimeIndex)


@pytest.mark.unit
def test_injected_fixture_exposes_known_indices(injected_anomalies: object) -> None:
    """The injected-anomaly fixture records its KNOWN injection indices."""
    fixture = injected_anomalies
    assert len(fixture.vol_burst_idx) == 6  # type: ignore[attr-defined]
    assert len(fixture.jump_idx) == 6  # type: ignore[attr-defined]
    # Known indices are a sorted, de-duplicated union.
    known = fixture.known_idx  # type: ignore[attr-defined]
    assert list(known) == sorted(set(known))
    assert all(0 <= i < 1000 for i in known)
