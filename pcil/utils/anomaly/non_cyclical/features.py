"""
Non-cyclical anomaly pipeline — Step 2: per-window feature extraction
======================================================================
Turn each fixed-length window into a fixed-size feature vector. Acoustic
data has 4 channels (Acceleration 0/1/2 + AE) — extract features per
channel and concatenate.

Pick ONE feature set:
  - Time-domain: RMS, peak, std, kurtosis, crest factor (per channel)
  - Frequency-domain: FFT band energies (e.g. 0–1 kHz, 1–5 kHz, 5–10 kHz) per channel
  - Both concatenated

Document your choice + which features in non_cyclical/README.md.
"""

from __future__ import annotations

import pandas as pd

CHANNEL_COLUMNS = [
    "Acceleration 0 (g)",
    "Acceleration 1 (g)",
    "Acceleration 2 (g)",
    "AE (V) (V)",
]


def extract_features(
    window_df: pd.DataFrame,
    *,
    channel_columns: list[str] = CHANNEL_COLUMNS,
) -> dict[str, float]:
    """
    Return a dict mapping feature_name -> float for one window.

    Naming convention: include the channel name, e.g.
        "accel_0_rms": ..., "ae_fft_band_0_1khz": ...

    Parameters
    ----------
    window_df : pd.DataFrame
        One window's rows (already sliced upstream).
    channel_columns : list[str]
        Which columns to extract features from.

    Returns
    -------
    dict[str, float]
        Same keys for every window (so downstream stacks correctly).

    TODO (teammate):
      1. Pick a feature set.
      2. For each channel in channel_columns, compute and add features
         with channel-prefixed names.
      3. NaN-safe.
    """
    raise NotImplementedError("TODO: pick and implement feature extraction.")


def stack_features(per_window_dicts: list[dict[str, float]]) -> pd.DataFrame:
    """Stack per-window feature dicts into one DataFrame (one row per window)."""
    return pd.DataFrame(per_window_dicts)
