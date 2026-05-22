"""
Cyclical anomaly pipeline — Step 0: data prep
==============================================
Turn `data/Clean_Data.csv` into `cyclical_dataset.csv` (training) and
`cyclical_eval.csv` (held-out with synthetic anomalies + cycle_label).

Steps (per the menu task 1):
  1. Reformat: semicolon -> comma; rename _time -> timestamp,
     SetPressure -> signal_value. Drop SetVelo.
  2. Optional: fake a second machine via timestamp shift +
     small constant offset to signal_value.
  3. Hold out the last 20% per machine -> eval set; inject 5–10%
     synthetic anomalies into fixed 1-second windows; label them.
  4. Remaining 80% -> training set (no labels).

Run from PCIL_dev/:
    python -m pcil.utils.anomaly.cyclical.prepare_data \\
        --input ../data/Clean_Data.csv \\
        --output-dir ../data/

Produces:
    <output-dir>/cyclical_dataset.csv
    <output-dir>/cyclical_eval.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
# Step 1 — reformat
# ─────────────────────────────────────────────────────────────

def reformat(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean up the raw `Clean_Data.csv`:
      - Rename `_time` -> `timestamp`, `SetPressure` -> `signal_value`.
      - Drop `SetVelo` (mostly constant in the recording).
      - Make sure `timestamp` is parsed as datetime.
    """

    df = raw.rename(columns={"_time": "timestamp", "SetPressure": "signal_value"})
    df = df.drop(columns=["SetVelo"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    return df


# ─────────────────────────────────────────────────────────────
# Step 2 — optional second machine
# ─────────────────────────────────────────────────────────────

def fake_second_machine(
    df: pd.DataFrame,
    *,
    offset: float = 0.05,
    time_shift: pd.Timedelta = pd.Timedelta(minutes=5),
) -> pd.DataFrame:
    """
    Duplicate rows with shifted time + nudged amplitude to test that the
    pipeline can carry separate machine baselines. Tag original as
    `inkjet_01`, duplicate as `inkjet_02`. Adds a `machine_id` column.

    This is only a local demo/stress test. In production, train a fitted
    anomaly bundle from real baseline data for each actual machine.
    """
    df = df.copy()
    df["machine_id"] = "inkjet_01"

    duplicate = df.copy()
    duplicate["timestamp"] += time_shift
    duplicate["signal_value"] += offset
    duplicate["machine_id"] = "inkjet_02"

    return duplicate


# ─────────────────────────────────────────────────────────────
# Step 3 — inject synthetic anomalies for eval
# ─────────────────────────────────────────────────────────────

def inject_anomalies(
    eval_df: pd.DataFrame,
    *,
    fraction: float = 0.08,
    window_size_rows: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Split `eval_df` into fixed `window_size_rows` windows, distort
    `fraction` of them at random, and add a `cycle_label` column
    (`normal` / `anomalous`) repeated across rows in each window.

    Vary the distortion type so the detector has multiple modes to catch:
      - amplitude *= 2
      - peak clipped at a low value
      - replaced with random noise
      - period stretched / compressed

    """
    if "signal_value" not in eval_df.columns:
        raise KeyError("Missing required column: signal_value")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0.0 and 1.0")
    if window_size_rows <= 0:
        raise ValueError("window_size_rows must be positive")

    eval_df = eval_df.copy()
    eval_df["cycle_label"] = "normal"

    n_rows = len(eval_df)
    if n_rows == 0 or fraction == 0.0:
        return eval_df

    rng = np.random.default_rng(seed)
    window_ids = np.arange(n_rows) // window_size_rows
    unique_windows = np.unique(window_ids)
    n_anomalous = int(np.ceil(len(unique_windows) * fraction))
    n_anomalous = min(len(unique_windows), max(1, n_anomalous))
    anomalous_windows = rng.choice(unique_windows, size=n_anomalous, replace=False)

    signal = eval_df["signal_value"].to_numpy(dtype=float, copy=True)
    global_std = float(np.nanstd(signal)) or 1.0
    global_min = float(np.nanmin(signal)) if not np.all(np.isnan(signal)) else 0.0

    for window_id in anomalous_windows:
        row_positions = np.flatnonzero(window_ids == window_id)
        if row_positions.size == 0:
            continue

        original = signal[row_positions].copy()
        distortion = rng.choice(["amplitude", "clip", "noise", "warp"])

        if distortion == "amplitude":
            distorted = original * 2.0
        elif distortion == "clip":
            low_clip = float(np.nanpercentile(original, 35))
            if not np.isfinite(low_clip):
                low_clip = global_min
            distorted = np.minimum(original, low_clip)
        elif distortion == "noise":
            centre = float(np.nanmean(original))
            if not np.isfinite(centre):
                centre = 0.0
            distorted = rng.normal(loc=centre, scale=global_std, size=row_positions.size)
        else:
            src = np.arange(row_positions.size, dtype=float)
            factor = float(rng.choice([0.65, 1.5]))
            warped_src = np.clip(src * factor, 0, row_positions.size - 1)
            distorted = np.interp(warped_src, src, np.nan_to_num(original, nan=global_min))

        signal[row_positions] = distorted
        eval_df.iloc[row_positions, eval_df.columns.get_loc("cycle_label")] = "anomalous"

    eval_df["signal_value"] = signal
    return eval_df


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prep cyclical training + eval datasets from Clean_Data.csv.",
    )
    parser.add_argument("--input", required=True, help="Path to Clean_Data.csv.")
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write cyclical_dataset.csv + cyclical_eval.csv to.",
    )
    parser.add_argument(
        "--no-fake-machine", action="store_true",
        help="Skip step 2 (don't fake a second machine).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for anomaly injection.")
    args = parser.parse_args()

    raw = pd.read_csv(args.input, sep=";")
    cleaned = reformat(raw)

    if args.no_fake_machine:
        cleaned["machine_id"] = "inkjet_01"
    else:
        cleaned = fake_second_machine(cleaned)

    # Hold out the last 20% per machine for eval
    train_frames, eval_frames = [], []
    for mid, group in cleaned.groupby("machine_id"):
        n = len(group)
        cutoff = int(n * 0.8)
        train_frames.append(group.iloc[:cutoff])
        eval_frames.append(group.iloc[cutoff:])
    train_df = pd.concat(train_frames, ignore_index=True)
    eval_df = pd.concat(eval_frames, ignore_index=True)
    eval_df = inject_anomalies(eval_df, seed=args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "cyclical_dataset.csv"
    eval_path = out_dir / "cyclical_eval.csv"
    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)

    print(f"Saved -> {train_path} ({len(train_df)} rows)")
    n_anomalous = int((eval_df["cycle_label"] == "anomalous").sum())
    print(f"Saved -> {eval_path} ({len(eval_df)} rows, {n_anomalous} anomalous)")


if __name__ == "__main__":
    main()
