"""
Cyclical anomaly pipeline — Step 4: model architecture + training
==================================================================
Pick ONE candidate. All four implement the same `.fit(X)` / `.score(X)`
interface so train.py and score.py work the same way regardless of
which one is chosen.

Candidates
----------
  ZScoreModel          — pure statistics, no training. Good baseline.
  IsolationForestModel — sklearn-native, light. Good first NN-free choice.
  OneClassSVMModel     — classic anomaly detector.
  AutoencoderModel     — heaviest, most flexible.

Spec each pick must include in cyclical/README.md
-------------------------------------------------
  - loss function (or "n/a" for IF / OCSVM)
  - optimiser (NN only)
  - stopping criterion
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class AnomalyModel(Protocol):
    """The interface every candidate satisfies."""

    def fit(self, X: np.ndarray) -> "AnomalyModel": ...
    def score(self, X: np.ndarray) -> np.ndarray: ...


# ─────────────────────────────────────────────────────────────
# Candidate 1: Z-score baseline (no training, pure statistics)
# ─────────────────────────────────────────────────────────────

class ZScoreModel:
    """
    Flag rows whose maximum absolute z-score across features exceeds
    `threshold`. "Trains" by storing the training set's per-feature
    mean and std.

    The fitted PerMachineNormaliser inside this machine's bundle already
    normalises features before this model sees them. This model performs
    a second-level aggregation that turns many normalised features into
    one anomaly score.
    """

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.mu_: np.ndarray | None = None
        self.sigma_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "ZScoreModel":
        """TODO (teammate): self.mu_, self.sigma_ = X.mean(axis=0), X.std(axis=0); guard against zero std."""
        raise NotImplementedError("TODO: ZScoreModel.fit")

    def score(self, X: np.ndarray) -> np.ndarray:
        """TODO (teammate): return per-row max |(X - mu) / sigma| as the anomaly score."""
        raise NotImplementedError("TODO: ZScoreModel.score")


# ─────────────────────────────────────────────────────────────
# Candidate 2: Isolation Forest (sklearn)
# ─────────────────────────────────────────────────────────────

class IsolationForestModel:
    """
    Wraps sklearn.ensemble.IsolationForest. Higher `score` = more anomalous.
    """

    def __init__(self, **kwargs):
        from sklearn.ensemble import IsolationForest
        self._model = IsolationForest(**kwargs)

    def fit(self, X: np.ndarray) -> "IsolationForestModel":
        """TODO (teammate): self._model.fit(X); return self."""
        raise NotImplementedError("TODO: IsolationForestModel.fit")

    def score(self, X: np.ndarray) -> np.ndarray:
        """TODO (teammate): return -self._model.score_samples(X) so higher = more anomalous."""
        raise NotImplementedError("TODO: IsolationForestModel.score")


# ─────────────────────────────────────────────────────────────
# Candidate 3: One-class SVM (sklearn)
# ─────────────────────────────────────────────────────────────

class OneClassSVMModel:
    """Wraps sklearn.svm.OneClassSVM. Higher `score` = more anomalous."""

    def __init__(self, **kwargs):
        from sklearn.svm import OneClassSVM
        self._model = OneClassSVM(**kwargs)

    def fit(self, X: np.ndarray) -> "OneClassSVMModel":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got shape {X.shape}.")

        self._model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return -self._model.score_samples(X) so higher = more anomalous."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got shape {X.shape}.")

        return -self._model.score_samples(X)


# ─────────────────────────────────────────────────────────────
# Candidate 4: Autoencoder (sketch — pick a NN library)
# ─────────────────────────────────────────────────────────────

class AutoencoderModel:
    """
    Reconstruction-error anomaly detector.

    Spec MUST be written in cyclical/README.md:
      - layers + bottleneck size
      - activation functions
      - loss function (e.g. MSE)
      - optimiser (e.g. Adam, lr=1e-3)
      - stopping criterion (e.g. early stop on val MSE plateau)
      - choice of NN library (PyTorch / Keras / sklearn MLPRegressor wrapper)
    """

    def __init__(self, **kwargs):
        from sklearn.neural_network import MLPRegressor

        hidden_layer_sizes = kwargs.pop("hidden_layer_sizes", (64, 16, 64))
        activation = kwargs.pop("activation", "relu")
        solver = kwargs.pop("solver", "adam")
        learning_rate_init = kwargs.pop("learning_rate_init", 1e-3)
        max_iter = kwargs.pop("max_iter", 200)
        early_stopping = kwargs.pop("early_stopping", True)
        n_iter_no_change = kwargs.pop("n_iter_no_change", 10)
        validation_fraction = kwargs.pop("validation_fraction", 0.1)
        random_state = kwargs.pop("random_state", 42)

        self._model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            n_iter_no_change=n_iter_no_change,
            validation_fraction=validation_fraction,
            random_state=random_state,
            **kwargs,
        )

    def fit(self, X: np.ndarray) -> "AutoencoderModel":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got shape {X.shape}.")

        self._n_features_in_ = X.shape[1]
        self._model.fit(X, X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return per-row reconstruction error (e.g. MSE)."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got shape {X.shape}.")
        if getattr(self, "_n_features_in_", X.shape[1]) != X.shape[1]:
            raise ValueError(
                f"Expected {self._n_features_in_} features, got {X.shape[1]}."
            )

        reconstructed = self._model.predict(X)
        return np.mean((X - reconstructed) ** 2, axis=1)
