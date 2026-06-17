"""Isolation Forest anomaly detector (Liu, Ting & Zhou, 2008).

A thin, leakage-safe wrapper over :class:`sklearn.ensemble.IsolationForest`.
The forest is fitted on the TRAIN feature slice ONLY; the per-day anomaly score
is ``-score_samples`` (so that higher = more anomalous, matching the
autoencoder's sign convention), and the flag threshold is the
``1 - contamination`` quantile of the TRAIN scores. The fitted forest then
scores the disjoint OOS slice without ever seeing it during fit.

FULL-SAMPLE-LEAKAGE GUARD: ``fit`` consumes only the train slice; ``score``
transforms the OOS slice with the already-fitted estimator and the
train-derived threshold. Fitting on the full sample is look-ahead and is the
top project risk.

scikit-learn is imported LAZILY inside the methods so importing this module has
no side effects and does not require scikit-learn.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from anomaly_detector.detectors.result import AnomalyResult

#: Default contamination (expected anomalous fraction) for the flag threshold.
DEFAULT_CONTAMINATION: float = 0.02

#: Default number of base estimators in the forest.
DEFAULT_N_ESTIMATORS: int = 200


class IsolationForestDetector:
    """Leakage-safe Isolation Forest anomaly detector.

    Wraps :class:`sklearn.ensemble.IsolationForest` with a fit-on-train /
    score-on-OOS discipline and a train-derived flag threshold. The anomaly
    score is ``-score_samples`` so that larger scores mean more anomalous.

    Parameters
    ----------
    contamination:
        Expected anomalous fraction; sets the flag threshold as the
        ``1 - contamination`` quantile of the TRAIN scores. Must satisfy
        ``0 < contamination < 0.5``.
    n_estimators:
        Number of base trees in the forest.
    seed:
        Master RNG seed (feeds the estimator's ``random_state`` deterministically
        via :func:`anomaly_detector._rng.make_rng`).
    """

    def __init__(
        self,
        *,
        contamination: float = DEFAULT_CONTAMINATION,
        n_estimators: int = DEFAULT_N_ESTIMATORS,
        seed: int = 7,
    ) -> None:
        raise NotImplementedError

    def fit(self, train_features: pd.DataFrame) -> IsolationForestDetector:
        """Fit the forest and the flag threshold on the TRAIN slice ONLY.

        Standardizes ``train_features`` (a scaler fitted on the train slice),
        fits the Isolation Forest on the standardized train slice, then sets the
        flag threshold to the ``1 - contamination`` quantile of the train scores.
        The OOS slice is never touched here.

        Parameters
        ----------
        train_features:
            The per-day TRAIN feature matrix (rows = date, columns = feature).

        Returns
        -------
        IsolationForestDetector
            ``self``, fitted (enables method chaining).

        Raises
        ------
        NotImplementedError
            Always — this is a typed stub for parallel authoring.
        """
        raise NotImplementedError

    def score(self, test_features: pd.DataFrame) -> AnomalyResult:
        """Score the disjoint OOS slice with the train-fitted forest.

        Applies the train-fitted scaler and forest to ``test_features``,
        computes ``-score_samples`` per day, and flags days whose score exceeds
        the train-derived threshold.

        Parameters
        ----------
        test_features:
            The per-day OOS feature matrix (disjoint from the train slice).

        Returns
        -------
        AnomalyResult
            The OOS scores, flags, and the train-derived threshold.

        Raises
        ------
        NotFittedError
            If :meth:`fit` has not been called.
        NotImplementedError
            Always — this is a typed stub for parallel authoring.
        """
        raise NotImplementedError

    def raw_score_samples(self, features: pd.DataFrame) -> np.ndarray:
        """Return raw sklearn ``score_samples`` for parity testing.

        Exposes the underlying estimator's ``score_samples`` (the un-negated
        sklearn quantity) so the parity suite can assert agreement with a raw
        ``IsolationForest`` to ``1e-10``.

        Parameters
        ----------
        features:
            A per-day feature matrix to score.

        Returns
        -------
        numpy.ndarray
            The raw ``score_samples`` values.

        Raises
        ------
        NotImplementedError
            Always — this is a typed stub for parallel authoring.
        """
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable description of the detector configuration.

        Raises
        ------
        NotImplementedError
            Always — this is a typed stub for parallel authoring.
        """
        raise NotImplementedError
