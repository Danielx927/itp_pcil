"""
Non-cyclical anomaly pipeline — score CLI
==========================================
Takes an acoustic CSV + that machine's trained .pkl, runs window
slicing, feature extraction, and the fitted per-machine normaliser, then
returns one row per window with an `anomaly_score` column.

Run from PCIL_dev/:
    python -m pcil.utils.anomaly.non_cyclical.score \\
        --input "../data/Inkjet Printer Data Collection/Acoustic Sensor Data/machine_on_anomaly.csv" \\
        --model ../data/non_cyclical_inkjet_01.pkl \\
        --output ../data/non_cyclical_anomaly_scored.csv

For evaluation, run score on both *_clean.csv and *_anomaly.csv files
and use filename labels as ground truth for precision / recall.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from pcil.utils.anomaly.non_cyclical.slice import detect_windows
from pcil.utils.anomaly.non_cyclical.features import extract_features, stack_features
from pcil.utils.anomaly.non_cyclical.train import load_acoustic


def score(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Return a per-window DataFrame with features + anomaly_score."""
    machine_id = bundle["machine_id"]
    machine_id_column = bundle["machine_id_column"]
    window_size_rows = bundle["window_size_rows"]
    channel_columns = bundle["channel_columns"]
    feature_columns = bundle["feature_columns"]
    normaliser = bundle["normaliser"]
    model = bundle["model"]

    df_sorted = df.reset_index(drop=True)
    rows = []
    for start, end in detect_windows(df_sorted, window_size_rows=window_size_rows):
        features = extract_features(df_sorted.iloc[start:end], channel_columns=channel_columns)
        features[machine_id_column] = machine_id
        features["window_start_idx"] = start
        rows.append(features)

    feature_df = stack_features(rows)
    feature_df_norm = normaliser.transform(feature_df, machine_id_column=machine_id_column)
    X = feature_df_norm[feature_columns].to_numpy(dtype=float)
    feature_df["anomaly_score"] = model.score(X)
    return feature_df


def main():
    parser = argparse.ArgumentParser(description="Score non-cyclical data with a trained model.")
    parser.add_argument("--input", required=True, help="Acoustic CSV to score.")
    parser.add_argument("--model", required=True, help="Path to .pkl from train.py.")
    parser.add_argument("--output", required=True, help="Where to save the scored CSV.")
    parser.add_argument("--header-skiprows", type=int, default=5,
                        help="Metadata lines to skip in the CSV.")
    args = parser.parse_args()

    df = load_acoustic(Path(args.input), header_skiprows=args.header_skiprows)
    bundle = joblib.load(args.model)
    scored = score(df, bundle)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    print(f"Scored {len(scored)} windows.")


if __name__ == "__main__":
    main()
