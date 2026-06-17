"""PCA-reconstruction-error autoencoder anomaly detector (Sakurada & Yairi, 2014).

A near-zero-dependency "autoencoder": instead of a neural network, it fits a
:class:`sklearn.decomposition.PCA` on the standardized TRAIN feature slice and
treats the per-day reconstruction error as the anomaly score::

    score(x) = || x - pca.inverse_transform(pca.transform(x)) ||^2

A day reconstructs poorly (high score) when it lies off the principal subspace
learned from the calm TRAIN data — i.e. it is anomalous. The flag threshold is a
quantile of the TRAIN reconstruction errors. There is NO torch / tensorflow.

FULL-SAMPLE-LEAKAGE GUARD: the scaler, the PCA basis, AND the error-quantile
threshold are all fitted on the TRAIN slice ONLY; ``score`` projects the
disjoint OOS slice onto the frozen basis. Fitting PCA on the full sample is
look-ahead and is the top project risk.

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

#: Default train-error quantile used as the reconstruction-error flag threshold.
DEFAULT_CONTAMINATION: float = 0.02

#: Default number of principal components retained (the bottleneck width).
DEFAULT_N_COMPONENTS: int = 3


class PCAAutoencoderDetector:
    """Leakage-safe PCA reconstruction-error autoencoder.

    Fits a PCA bottleneck on the standardized TRAIN slice; the per-day anomaly
    score is the squared reconstruction error, and the flag threshold is the
    ``1 - contamination`` quantile of the TRAIN reconstruction errors.

    Parameters
    ----------
    n_components:
        Number of principal components retained (the autoencoder bottleneck
        width). Must be ``>= 1`` and ``<= n_features``.
    contamination:
        Train-error quantile for the flag threshold; sets the threshold as the
        ``1 - contamination`` quantile of the TRAIN reconstruction errors. Must
        satisfy ``0 < contamination < 0.5``.
    seed:
        Master RNG seed (feeds the PCA ``random_state`` deterministically via
        :func:`anomaly_detector._rng.make_rng`).
    """

    def __init__(
        self,
        *,
        n_components: int = DEFAULT_N_COMPONENTS,
        contamination: float = DEFAULT_CONTAMINATION,
        seed: int = 7,
    ) -> None:
        raise NotImplementedError

    def fit(self, train_features: pd.DataFrame) -> PCAAutoencoderDetector:
        """Fit the scaler, PCA basis, and error threshold on the TRAIN slice ONLY.

        Standardizes ``train_features`` (scaler fitted on the train slice), fits
        the PCA on the standardized train slice, computes the train
        reconstruction errors, and sets the threshold to their
        ``1 - contamination`` quantile. The OOS slice is never touched here.

        Parameters
        ----------
        train_features:
            The per-day TRAIN feature matrix (rows = date, columns = feature).

        Returns
        -------
        PCAAutoencoderDetector
            ``self``, fitted (enables method chaining).

        Raises
        ------
        NotImplementedError
            Always — this is a typed stub for parallel authoring.
        """
        raise NotImplementedError

    def score(self, test_features: pd.DataFrame) -> AnomalyResult:
        """Score the disjoint OOS slice with the train-fitted PCA basis.

        Standardizes ``test_features`` with the train scaler, projects onto and
        reconstructs from the frozen PCA basis, computes the squared
        reconstruction error per day, and flags days whose error exceeds the
        train-derived threshold.

        Parameters
        ----------
        test_features:
            The per-day OOS feature matrix (disjoint from the train slice).

        Returns
        -------
        AnomalyResult
            The OOS reconstruction-error scores, flags, and threshold.

        Raises
        ------
        NotFittedError
            If :meth:`fit` has not been called.
        NotImplementedError
            Always — this is a typed stub for parallel authoring.
        """
        raise NotImplementedError

    def reconstruction_error(self, features: pd.DataFrame) -> np.ndarray:
        """Return the per-day squared reconstruction error for parity testing.

        Exposes ``|| x - pca.inverse_transform(pca.transform(x)) ||^2`` computed
        on the (train-)scaled ``features`` so the parity suite can assert
        agreement with a raw sklearn PCA ``inverse_transform`` to ``1e-10``.

        Parameters
        ----------
        features:
            A per-day feature matrix to reconstruct.

        Returns
        -------
        numpy.ndarray
            The per-day squared reconstruction errors.

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
