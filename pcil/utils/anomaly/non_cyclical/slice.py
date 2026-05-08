"""
Non-cyclical anomaly pipeline — Step 1: fixed window slicing
=============================================================
Non-cyclical data (vibration RMS, acoustic emission, temperature)
doesn't have natural cycle boundaries — chop into fixed-length windows.

Pick ONE window length with reasoning:
  - 0.1 s @ 25.6 kHz = 2560 samples per window
  - 0.5 s @ 25.6 kHz = 12800 samples per window
  - 1.0 s @ 25.6 kHz = 25600 samples per window

Document your choice in non_cyclical/README.md.
"""

from __future__ import annotations

from typing import Iterator

import pandas as pd


def detect_windows(
    df: pd.DataFrame,
    *,
    window_size_rows: int,
    stride: int | None = None,
) -> Iterator[tuple[int, int]]:
    """
    Yield (start_idx, end_idx) tuples — one per fixed-length window.

    Parameters
    ----------
    df : pd.DataFrame
        Sorted ascending. Single-machine slice.
    window_size_rows : int
        Number of rows per window.
    stride : int | None
        Step between window starts. None means non-overlapping
        (stride = window_size_rows).

    Yields
    ------
    (start_idx, end_idx) : tuple[int, int]
        Half-open: df.iloc[start_idx:end_idx] is one window.

    TODO (teammate):
      1. Decide on stride default (non-overlapping is simplest).
      2. Iterate from 0 to len(df) - window_size_rows in steps of `stride`.
      3. Edge case: drop the trailing partial window or pad it.
    """
    raise NotImplementedError("TODO: implement detect_windows")
