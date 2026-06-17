"""Detector-agreement and proxy-label diagnostics.

The HEADLINE of this tool is descriptive, not predictive: how much do the two
independent detectors agree, and how well do their flags line up with a
TRANSPARENT proxy label and a set of KNOWN macro-stress windows? There is no
ground-truth anomaly label, so nothing here implies a tradable signal.

Quantities computed
-------------------
- Jaccard / overlap between the two detectors' OOS flag sets (the primary
  stability metric — expected to be MODEST, ~0.3-0.5).
- Regime alignment: overlap of each detector's flags with KNOWN stress windows
  (e.g. 2020-03, 2018-Q4, the 2022 selloff) or a VIX-spike proxy.
- Precision / recall of the flags against a transparent proxy label
  ``|causal z-return| > 3`` (expected LOW precision — flags are diagnostic).

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

#: Z-score cutoff for the transparent proxy anomaly label ``|z-return| > Z``.
PROXY_Z_THRESHOLD: float = 3.0

#: Known macro-stress windows (inclusive ISO date ranges) used for regime
#: alignment. These are TRANSPARENT, well-documented historical stress periods,
#: NOT a fitted ground-truth label.
KNOWN_STRESS_WINDOWS: tuple[tuple[str, str], ...] = (
    ("2018-10-01", "2018-12-31"),  # 2018-Q4 selloff
    ("2020-02-20", "2020-04-30"),  # COVID crash
    ("2022-01-01", "2022-10-31"),  # 2022 rate-driven selloff
)


@dataclass(frozen=True, slots=True)
class AgreementResult:
    """Immutable descriptive-agreement summary for a pair of detectors.

    Attributes
    ----------
    jaccard:
        Jaccard index of the two detectors' OOS flag sets
        ``|A and B| / |A or B|`` (the primary stability metric).
    overlap_count:
        Number of OOS days flagged by BOTH detectors (``|A and B|``).
    n_flags_a:
        Number of OOS days flagged by detector A.
    n_flags_b:
        Number of OOS days flagged by detector B.
    proxy_precision:
        Precision of the (intersection or union) flags against the
        ``|z-return| > PROXY_Z_THRESHOLD`` proxy label (expected LOW).
    proxy_recall:
        Recall of those flags against the same proxy label.
    regime_alignment:
        Fraction of flagged days that fall inside a KNOWN stress window.
    top_anomaly_dates:
        The most anomalous OOS dates (ISO strings), agreed by both detectors
        where possible, for the summary block.
    meta:
        Free-form JSON-serializable provenance.
    """

    jaccard: float
    overlap_count: int
    n_flags_a: int
    n_flags_b: int
    proxy_precision: float
    proxy_recall: float
    regime_alignment: float
    top_anomaly_dates: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this summary.

        Raises
        ------
        NotImplementedError
            Always — this is a typed stub for parallel authoring.
        """
        raise NotImplementedError


def jaccard_index(flags_a: pd.Series, flags_b: pd.Series) -> float:
    """Jaccard index of two boolean OOS flag series.

    Returns ``|A and B| / |A or B|`` over the aligned dates, where ``A``/``B``
    are the sets of dates each detector flagged. Defined as ``0.0`` when neither
    detector flags anything (empty union).

    Parameters
    ----------
    flags_a, flags_b:
        Boolean per-day flag series, aligned (or alignable) on their date index.

    Returns
    -------
    float
        The Jaccard index in ``[0, 1]``.

    Raises
    ------
    NotImplementedError
        Always — this is a typed stub for parallel authoring.
    """
    raise NotImplementedError


def proxy_precision_recall(
    flags: pd.Series,
    returns: pd.Series,
    *,
    window: int,
    z_threshold: float = PROXY_Z_THRESHOLD,
) -> tuple[float, float]:
    """Precision/recall of ``flags`` against the ``|causal z-return| > z`` proxy.

    The proxy label is TRANSPARENT and causal (built from
    :func:`anomaly_detector.features.engineer.causal_return_zscore`), NOT a
    ground-truth anomaly label. Low precision is the honest expected outcome.

    Parameters
    ----------
    flags:
        Boolean per-day flag series over the OOS slice.
    returns:
        The per-day return series over the same span (for the proxy label).
    window:
        Rolling window for the causal z-score.
    z_threshold:
        Absolute z-score cutoff defining the proxy positive class.

    Returns
    -------
    tuple[float, float]
        ``(precision, recall)`` of the flags against the proxy label.

    Raises
    ------
    NotImplementedError
        Always — this is a typed stub for parallel authoring.
    """
    raise NotImplementedError


def regime_alignment(
    flags: pd.Series,
    *,
    windows: tuple[tuple[str, str], ...] = KNOWN_STRESS_WINDOWS,
) -> float:
    """Fraction of flagged days that fall inside a KNOWN stress window.

    Parameters
    ----------
    flags:
        Boolean per-day flag series over the OOS slice.
    windows:
        Inclusive ISO date ranges of known macro-stress periods.

    Returns
    -------
    float
        The fraction of flagged days inside any window (``0.0`` if no flags).

    Raises
    ------
    NotImplementedError
        Always — this is a typed stub for parallel authoring.
    """
    raise NotImplementedError


def compute_agreement(
    result_a: Any,
    result_b: Any,
    returns: pd.Series,
    *,
    window: int,
) -> AgreementResult:
    """Assemble the full descriptive-agreement summary for two detector results.

    Combines the Jaccard index, proxy precision/recall, and regime alignment
    into a single :class:`AgreementResult` for the API summary block.

    Parameters
    ----------
    result_a, result_b:
        The two detectors' :class:`~anomaly_detector.detectors.result.AnomalyResult`
        objects over the SAME OOS slice.
    returns:
        The per-day return series over the OOS slice (for the proxy label).
    window:
        Rolling window for the causal z-score proxy.

    Returns
    -------
    AgreementResult
        The descriptive agreement summary.

    Raises
    ------
    NotImplementedError
        Always — this is a typed stub for parallel authoring.
    """
    raise NotImplementedError
