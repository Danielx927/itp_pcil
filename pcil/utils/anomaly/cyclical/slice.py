"""
Cyclical anomaly pipeline — Step 1: cycle detection
====================================================
Cuts a continuous cyclical signal into one cycle per unit so the next
step can build a fixed-size feature vector per cycle.

Pick ONE detection method:
  - peak detection (scipy.signal.find_peaks on the signal)
  - zero-crossing detection (sign changes of the detrended signal)
  - fixed-period window (if cycle period is known a priori)

Document your choice + rationale in cyclical/README.md.
"""

from __future__ import annotations

from typing import Iterator

import pandas as pd


def detect_cycles(
    df: pd.DataFrame,
    *,
    signal_column: str = "signal_value",
    timestamp_column: str = "timestamp",
) -> Iterator[tuple[int, int]]:
    """
    Yield (start_idx, end_idx) tuples — one per detected cycle.

    Parameters
    ----------
    df : pd.DataFrame
        Sorted by timestamp ascending. Single-machine slice.
    signal_column : str, default "signal_value"
    timestamp_column : str, default "timestamp"

    Yields
    ------
    (start_idx, end_idx) : tuple[int, int]
        Half-open: df.iloc[start_idx:end_idx] is one cycle.

    TODO (teammate):
      1. Pick a detection method (peak / zero-crossing / fixed-period).
      2. The Clean_Data.csv source has a 250 Hz sample artefact (values
         repeat in 4-row groups at the 1 kHz log rate). Decimate to
         every 4th row or smooth before peak detection.
      3. Yield each detected cycle.
      4. Edge cases: signal too short to contain a full cycle -> yield nothing.
    """
    raise NotImplementedError(
        "TODO: pick and implement a cycle detection strategy."
    )
