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

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid

# ── Constants ─────────────────────────────────────────────────────────────────

# Minimum rows for a valid cycle; shorter cycles return a zero-vector.
MIN_CYCLE_ROWS: int = 4

# Number of interpolated samples for the resampled-waveform method.
RESAMPLE_LENGTH: int = 100

# Number of FFT magnitude coefficients kept for the fft method.
N_FFT_BINS: int = 20

# Sampling interval in ms at 250 Hz (used when no timestamp column present).
_ROW_INTERVAL_MS: float = 4.0   # 1 / 250 Hz = 4 ms


def extract_features(
    cycle_df: pd.DataFrame,
    *,
    method: str = "summary",
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
    signal = cycle_df[signal_column].to_numpy(dtype=float)
    n = len(signal)

    # ── Method-specific zero fallback ─────────────────────────────────────────
    if method == "summary":
        zero = {
            "peak": 0.0, "trough": 0.0, "mean": 0.0,
            "std": 0.0, "integrated_area": 0.0, "cycle_duration_ms": 0.0,
        }
    elif method == "resampled-waveform":
        zero = {f"s_{i:03d}": 0.0 for i in range(RESAMPLE_LENGTH)}
    elif method == "fft":
        zero = {f"fft_bin_{i:03d}": 0.0 for i in range(N_FFT_BINS)}
    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            "Choose one of: summary, resampled-waveform, fft"
        )

    if n < MIN_CYCLE_ROWS or np.all(np.isnan(signal)):
        return zero

    # Replace residual NaNs with zero rather than propagating them
    signal = np.nan_to_num(signal, nan=0.0)

    # ── Shared: wall-clock duration ───────────────────────────────────────────
    if "timestamp" in cycle_df.columns:
        ts = pd.to_datetime(cycle_df["timestamp"], format="mixed", utc=True)
        duration_ms = (ts.iloc[-1] - ts.iloc[0]).total_seconds() * 1000.0
    else:
        duration_ms = float(n - 1) * _ROW_INTERVAL_MS

    # ── summary ───────────────────────────────────────────────────────────────
    if method == "summary":
        t = np.linspace(0.0, duration_ms, n)
        return {
            "peak":              float(np.max(signal)),
            "trough":            float(np.min(signal)),
            "mean":              float(np.mean(signal)),
            "std":               float(np.std(signal, ddof=1)) if n > 1 else 0.0,
            "integrated_area":   float(trapezoid(np.abs(signal), t)),
            "cycle_duration_ms": duration_ms,
        }

    # ── resampled-waveform ────────────────────────────────────────────────────
    elif method == "resampled-waveform":
        # Interpolate the cycle onto RESAMPLE_LENGTH evenly spaced positions.
        # np.interp maps fractional source indices to signal values, giving
        # every cycle a uniform shape regardless of its original row count.
        src_idx = np.linspace(0, n - 1, RESAMPLE_LENGTH)
        resampled = np.interp(src_idx, np.arange(n), signal)
        return {f"s_{i:03d}": float(v) for i, v in enumerate(resampled)}

    # ── fft ───────────────────────────────────────────────────────────────────
    elif method == "fft":
        # Mean-centre before FFT so DC bin reflects waveform shape only,
        # not the absolute pressure level (which differs between machines).
        sig_mean = float(np.mean(signal))
        fft_mags = np.abs(np.fft.rfft(signal - sig_mean))
        # Trim or zero-pad to exactly N_FFT_BINS coefficients so that the
        # feature vector length is fixed regardless of cycle length.
        coeffs = np.zeros(N_FFT_BINS)
        take = min(N_FFT_BINS, len(fft_mags))
        coeffs[:take] = fft_mags[:take]
        return {f"fft_bin_{i:03d}": float(v) for i, v in enumerate(coeffs)}


def stack_features(per_cycle_dicts: list[dict[str, float]]) -> pd.DataFrame:
    """Stack a list of per-cycle feature dicts into one DataFrame (one row per cycle)."""
    return pd.DataFrame(per_cycle_dicts)
