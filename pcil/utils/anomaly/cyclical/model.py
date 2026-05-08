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
        raise NotImplementedError("TODO: OneClassSVMModel.fit")

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return -self._model.score_samples(X) so higher = more anomalous."""
        raise NotImplementedError("TODO: OneClassSVMModel.score")


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
        # TODO (teammate): build the architecture per the README spec.
        self._kwargs = kwargs

    def fit(self, X: np.ndarray) -> "AutoencoderModel":
        raise NotImplementedError("TODO: AutoencoderModel.fit")

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return per-row reconstruction error (e.g. MSE)."""
        raise NotImplementedError("TODO: AutoencoderModel.score")
