"""
Cyclical anomaly pipeline — Step 2: per-cycle feature extraction
=================================================================
Turn each detected cycle into a fixed-size numeric feature vector.

Pick ONE feature set:
  - summary stats:  peak, trough, mean, std, integrated_area, cycle_duration_ms
  - resampled waveform: interpolate every cycle to a fixed length (e.g. 100 samples)
  - FFT band energies

Document your choice + which features you settled on in cyclical/README.md.
"""

from __future__ import annotations

import pandas as pd


def extract_features(
    cycle_df: pd.DataFrame,
    *,
    signal_column: str = "signal_value",
) -> dict[str, float]:
    """
    Return a dict mapping feature_name -> float for one cycle.

    Parameters
    ----------
    cycle_df : pd.DataFrame
        One cycle's worth of rows (already sliced upstream).
    signal_column : str

    Returns
    -------
    dict[str, float]
        IMPORTANT: every cycle MUST return the same set of keys, in the
        same order — downstream code stacks these into a feature matrix.

    TODO (teammate):
      1. Pick a feature set (see module docstring).
      2. Compute each feature; return as a dict with stable keys.
      3. NaN-safe: cycle_df might be very short if detection mis-fires.
         Decide what to return in that case (zeros? skip?).
    """
    raise NotImplementedError("TODO: pick and implement feature extraction.")


def stack_features(per_cycle_dicts: list[dict[str, float]]) -> pd.DataFrame:
    """Stack a list of per-cycle feature dicts into one DataFrame (one row per cycle)."""
    return pd.DataFrame(per_cycle_dicts)
