"""
Non-cyclical anomaly pipeline — Step 4: model architecture + training
======================================================================
Same candidates and interface as the cyclical pipeline. Pick ONE.

Candidates
----------
  ZScoreModel          — pure statistics, no training. Good baseline.
  IsolationForestModel — sklearn-native, light. Good first NN-free choice.
  OneClassSVMModel     — classic anomaly detector.
  AutoencoderModel     — heaviest, most flexible.

Spec each pick must include in non_cyclical/README.md
-----------------------------------------------------
  - loss function (or "n/a" for IF / OCSVM)
  - optimiser (NN only)
  - stopping criterion
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class AnomalyModel(Protocol):
    def fit(self, X: np.ndarray) -> "AnomalyModel": ...
    def score(self, X: np.ndarray) -> np.ndarray: ...


class ZScoreModel:
    """Per-feature mean/std at fit time; per-row max |z| at score time."""

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.mu_: np.ndarray | None = None
        self.sigma_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "ZScoreModel":
        raise NotImplementedError("TODO: ZScoreModel.fit")

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: ZScoreModel.score")


class IsolationForestModel:
    """sklearn IsolationForest wrapper. Higher score = more anomalous."""

    def __init__(self, **kwargs):
        from sklearn.ensemble import IsolationForest
        self._model = IsolationForest(**kwargs)

    def fit(self, X: np.ndarray) -> "IsolationForestModel":
        raise NotImplementedError("TODO: IsolationForestModel.fit")

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: IsolationForestModel.score")


class OneClassSVMModel:
    """sklearn OneClassSVM wrapper. Higher score = more anomalous."""

    def __init__(self, **kwargs):
        from sklearn.svm import OneClassSVM
        self._model = OneClassSVM(**kwargs)

    def fit(self, X: np.ndarray) -> "OneClassSVMModel":
        raise NotImplementedError("TODO: OneClassSVMModel.fit")

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: OneClassSVMModel.score")


class AutoencoderModel:
    """
    Reconstruction-error anomaly detector.

    Spec MUST be written in non_cyclical/README.md:
      - layers + bottleneck size
      - activation functions
      - loss function (e.g. MSE)
      - optimiser (e.g. Adam, lr=1e-3)
      - stopping criterion (e.g. early stop on val MSE plateau)
      - choice of NN library
    """

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def fit(self, X: np.ndarray) -> "AutoencoderModel":
        raise NotImplementedError("TODO: AutoencoderModel.fit")

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TODO: AutoencoderModel.score")
