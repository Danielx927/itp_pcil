"""
Cyclical anomaly pipeline — Step 1: cycle detection
====================================================
Cuts a continuous cyclical signal into one cycle per unit so the next
step can build a fixed-size feature vector per cycle.

Pick ONE detection method:
  - peak detection (local maxima on the smoothed signal)
  - zero-crossing detection (sign changes of the detrended signal)
  - fixed-period window (if cycle period is known a priori)

Document your choice + rationale in cyclical/README.md.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd


_DEFAULT_DECIMATION: int = 4
_DEFAULT_FIXED_PERIOD_ROWS: int = 1000
_MIN_CYCLE_ROWS: int = 4


def detect_cycles(
    df: pd.DataFrame,
    *,
    signal_column: str = "signal_value",
    timestamp_column: str = "timestamp",
    method: str = "peak",
    decimation: int = _DEFAULT_DECIMATION,
    fixed_period_rows: int = _DEFAULT_FIXED_PERIOD_ROWS,
    min_cycle_rows: int = _MIN_CYCLE_ROWS,
    peak_distance_rows: int | None = None,
    peak_prominence: float | None = None,
    fallback_to_fixed: bool = False,
) -> Iterator[tuple[int, int]]:
    """
    Yield (start_idx, end_idx) tuples — one per detected cycle.

    Parameters
    ----------
    df : pd.DataFrame
        Sorted by timestamp ascending. Single-machine slice.
    signal_column : str, default "signal_value"
    timestamp_column : str, default "timestamp"
    method : str, default "peak"
        One of "peak", "zero-crossing", or "fixed-period".
    decimation : int, default 4
        Keep every nth row for peak detection to counter the 1 kHz log /
        250 Hz source artefact described in the README.
    fixed_period_rows : int, default 1000
        Cycle length used when method="fixed-period".
    min_cycle_rows : int, default 4
        Drop detected spans shorter than this many rows.
    peak_distance_rows : int | None
        Optional minimum row distance between peaks.
    peak_prominence : float | None
        Optional minimum peak height above neighbouring samples. Defaults
        to a fraction of signal std.
    fallback_to_fixed : bool, default False
        If True, peak / zero-crossing methods fall back to fixed-period
        windows when too few boundaries are found.

    Yields
    ------
    (start_idx, end_idx) : tuple[int, int]
        Half-open: df.iloc[start_idx:end_idx] is one cycle.

    """
    if signal_column not in df.columns:
        raise KeyError(f"Missing signal column: {signal_column}")

    signal = _clean_signal(df[signal_column].to_numpy(dtype=float))
    n = len(signal)
    if n < min_cycle_rows:
        return

    if method == "peak":
        boundaries = _peak_boundaries(
            signal,
            decimation=decimation,
            min_cycle_rows=min_cycle_rows,
            peak_distance_rows=peak_distance_rows,
            peak_prominence=peak_prominence,
        )
    elif method == "zero-crossing":
        boundaries = _zero_crossing_boundaries(signal, min_cycle_rows=min_cycle_rows)
    elif method == "fixed-period":
        boundaries = range(0, n + 1, fixed_period_rows)
    else:
        raise ValueError(f"Unsupported method: {method}")

    boundaries = list(boundaries)
    if fallback_to_fixed and method != "fixed-period" and len(boundaries) < 2:
        boundaries = list(range(0, n + 1, fixed_period_rows))

    yield from _spans_from_boundaries(boundaries, n=n, min_cycle_rows=min_cycle_rows)


def _clean_signal(signal: np.ndarray) -> np.ndarray:
    """Return a finite float array suitable for boundary detection."""
    if signal.size == 0:
        return signal
    if np.all(np.isnan(signal)):
        return np.zeros_like(signal, dtype=float)

    fill = float(np.nanmedian(signal))
    return np.nan_to_num(signal, nan=fill, posinf=fill, neginf=fill)


def _peak_boundaries(
    signal: np.ndarray,
    *,
    decimation: int,
    min_cycle_rows: int,
    peak_distance_rows: int | None,
    peak_prominence: float | None,
) -> np.ndarray:
    decimation = max(1, int(decimation))
    reduced = signal[::decimation]
    if len(reduced) < 3:
        return np.array([], dtype=int)

    smoothed = _moving_average(reduced, width=11)
    default_distance_rows = max(min_cycle_rows, 800)
    distance = max(1, int((peak_distance_rows or default_distance_rows) / decimation))
    prominence = peak_prominence
    if prominence is None:
        std = float(np.std(smoothed))
        prominence = 0.10 * std if std > 0.0 else None

    peaks = _local_maxima(smoothed, distance=distance, prominence=prominence)
    return np.asarray(peaks, dtype=int) * decimation


def _zero_crossing_boundaries(
    signal: np.ndarray,
    *,
    min_cycle_rows: int,
) -> np.ndarray:
    detrended = signal - float(np.mean(signal))
    signs = np.signbit(detrended)
    crossings = np.flatnonzero(signs[1:] != signs[:-1]) + 1

    # Same-direction crossings mark full periods; adjacent crossings are
    # half-periods for a roughly sinusoidal signal.
    if len(crossings) >= 3:
        return crossings[::2]
    if len(crossings) >= 2 and crossings[-1] - crossings[0] >= min_cycle_rows:
        return crossings
    return np.array([], dtype=int)


def _moving_average(signal: np.ndarray, *, width: int) -> np.ndarray:
    if width <= 1 or len(signal) < width:
        return signal
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(signal, kernel, mode="same")


def _local_maxima(
    signal: np.ndarray,
    *,
    distance: int,
    prominence: float | None,
) -> np.ndarray:
    deltas = np.diff(signal)
    candidates = np.flatnonzero((deltas[:-1] > 0) & (deltas[1:] <= 0)) + 1
    if prominence is not None and candidates.size:
        min_height = float(np.median(signal)) + prominence
        candidates = candidates[signal[candidates] >= min_height]

    if candidates.size <= 1:
        return candidates

    kept: list[int] = []
    for idx in candidates:
        if not kept or idx - kept[-1] >= distance:
            kept.append(int(idx))
        elif signal[idx] > signal[kept[-1]]:
            kept[-1] = int(idx)
    return np.asarray(kept, dtype=int)


def _spans_from_boundaries(
    boundaries,
    *,
    n: int,
    min_cycle_rows: int,
) -> Iterator[tuple[int, int]]:
    last_start: int | None = None
    for boundary in boundaries:
        boundary = int(boundary)
        if boundary < 0 or boundary > n:
            continue
        if last_start is not None and boundary - last_start >= min_cycle_rows:
            yield last_start, boundary
        last_start = boundary
