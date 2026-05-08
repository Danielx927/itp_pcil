"""
Per-machine z-score normalisation
==================================
Winardi clarified on 2026-05-08 that the reusable deliverable is the
anomaly-detection pipeline definition, not one trained model instance
forced to work across every machine.

This normaliser is one preprocessing step inside that pipeline. For
each machine, fit the normaliser on that machine's own baseline/training
data, then train that machine's anomaly model instance on the normalised
features. When scoring new data from that same machine, reuse the fitted
normaliser and fitted model bundled in that machine's `.pkl`.

The shared part is the code / pipeline structure. The fitted baseline
statistics and trained model instance are machine-specific.

Usage
-----
    from pcil.utils.anomaly.normalise import PerMachineNormaliser

    normaliser = PerMachineNormaliser()
    normaliser.fit(train_df, machine_id_column="machine_id")
    z_scored = normaliser.transform(new_df, machine_id_column="machine_id")

Persisting:
    import joblib
    joblib.dump(normaliser, "normaliser.pkl")
"""

from __future__ import annotations

import pandas as pd


class PerMachineNormaliser:
    """
    Stores per-(machine, feature) mean and std at fit time. On transform,
    each row is z-scored against the baseline captured for that row's
    machine in this fitted normaliser instance.

    Unknown machines (not seen at fit time) raise. Use the fitted bundle
    for the correct machine, or train a new machine-specific instance.
    """

    def __init__(self):
        self.baselines_: dict[str, dict[str, tuple[float, float]]] = {}
        self.feature_columns_: list[str] = []

    def fit(
        self,
        df: pd.DataFrame,
        *,
        machine_id_column: str,
        feature_columns: list[str] | None = None,
    ) -> "PerMachineNormaliser":
        """Compute mean / std per (machine_id, feature) pair."""
        if feature_columns is None:
            feature_columns = [
                c for c in df.columns
                if c != machine_id_column and pd.api.types.is_numeric_dtype(df[c])
            ]
        self.feature_columns_ = list(feature_columns)
        self.baselines_ = {}
        for machine_id, group in df.groupby(machine_id_column):
            self.baselines_[machine_id] = {
                col: (
                    float(group[col].mean()),
                    float(group[col].std(ddof=0)) or 1.0,  # avoid /0 on flat columns
                )
                for col in self.feature_columns_
            }
        return self

    def transform(
        self,
        df: pd.DataFrame,
        *,
        machine_id_column: str,
    ) -> pd.DataFrame:
        """Return a copy of df with each feature column z-scored by its row's machine baseline."""
        if not self.baselines_:
            raise RuntimeError("PerMachineNormaliser must be fit() before transform().")

        unknown = set(df[machine_id_column].unique()) - set(self.baselines_)
        if unknown:
            raise KeyError(
                f"Unknown machine_id(s) {sorted(unknown)} — fit() didn't see them. "
                f"Use a bundle trained for that machine, or train a new "
                f"machine-specific normaliser/model bundle."
            )

        out = df.copy()
        for col in self.feature_columns_:
            if col not in out.columns:
                continue
            mu_series = out[machine_id_column].map(lambda m: self.baselines_[m][col][0])
            sigma_series = out[machine_id_column].map(lambda m: self.baselines_[m][col][1])
            out[col] = (out[col] - mu_series) / sigma_series
        return out

    def fit_transform(
        self,
        df: pd.DataFrame,
        *,
        machine_id_column: str,
        feature_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        return self.fit(
            df,
            machine_id_column=machine_id_column,
            feature_columns=feature_columns,
        ).transform(df, machine_id_column=machine_id_column)
