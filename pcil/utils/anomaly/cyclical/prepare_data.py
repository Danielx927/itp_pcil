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

    TODO (teammate):
      - Implement the rename + drop.
      - Return the cleaned frame.
    """
    raise NotImplementedError("TODO: reformat()")


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

    TODO (teammate):
      - Copy df, shift df_copy['timestamp'] by `time_shift`, add `offset`
        to df_copy['signal_value'], tag as inkjet_02.
      - Tag the original as inkjet_01.
      - Concatenate, sort by timestamp, return.
    """
    raise NotImplementedError("TODO: fake_second_machine()")


# ─────────────────────────────────────────────────────────────
# Step 3 — inject synthetic anomalies for eval
# ─────────────────────────────────────────────────────────────

def inject_anomalies(
    eval_df: pd.DataFrame,
    *,
    fraction: float = 0.08,
    window_size_rows: int = 1000,
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

    TODO (teammate):
      - Group rows into windows (use np.arange(len(df)) // window_size_rows).
      - Pick `fraction` of windows at random (set a seed for reproducibility).
      - Apply a randomly chosen distortion to each picked window's signal_value.
      - Add `cycle_label` column.
      - Return the modified frame.
    """
    raise NotImplementedError("TODO: inject_anomalies()")


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
    eval_df = inject_anomalies(eval_df)

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
