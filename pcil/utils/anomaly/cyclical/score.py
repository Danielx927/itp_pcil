"""
Cyclical anomaly pipeline — score CLI
======================================
Takes a CSV slice + that machine's trained .pkl from train.py, reapplies
the same slicing + feature extraction + fitted per-machine normaliser,
and returns a per-cycle DataFrame with an `anomaly_score` column.

Run from PCIL_dev/:
    python -m pcil.utils.anomaly.cyclical.score \\
        --input ../data/cyclical_eval.csv \\
        --model ../data/cyclical_inkjet_01.pkl \\
        --output ../data/cyclical_eval_scored.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from pcil.utils.anomaly.cyclical.slice import detect_cycles
from pcil.utils.anomaly.cyclical.features import extract_features, stack_features


def score(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """
    Return a DataFrame with one row per detected cycle, containing the
    extracted features plus an `anomaly_score` column. The cycle's
    starting timestamp is preserved as `cycle_start_timestamp`.

    The bundle is fitted state for the machine(s) it was trained on; it
    is not the reusable pipeline definition itself.
    """
    machine_id_column = bundle["machine_id_column"]
    signal_column = bundle["signal_column"]
    timestamp_column = bundle["timestamp_column"]
    feature_columns = bundle["feature_columns"]
    normaliser = bundle["normaliser"]
    model = bundle["model"]

    rows = []
    for machine_id, group in df.groupby(machine_id_column):
        group = group.sort_values(timestamp_column).reset_index(drop=True)
        for start, end in detect_cycles(
            group, signal_column=signal_column, timestamp_column=timestamp_column,
        ):
            features = extract_features(group.iloc[start:end], signal_column=signal_column)
            features[machine_id_column] = machine_id
            features["cycle_start_timestamp"] = group.iloc[start][timestamp_column]
            rows.append(features)

    feature_df = stack_features(rows)
    feature_df_norm = normaliser.transform(feature_df, machine_id_column=machine_id_column)
    X = feature_df_norm[feature_columns].to_numpy(dtype=float)
    feature_df["anomaly_score"] = model.score(X)
    return feature_df


def main():
    parser = argparse.ArgumentParser(description="Score cyclical data with a trained model.")
    parser.add_argument("--input", required=True, help="CSV slice to score.")
    parser.add_argument("--model", required=True, help="Path to .pkl from train.py.")
    parser.add_argument("--output", required=True, help="Where to save scored CSV.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    bundle = joblib.load(args.model)
    scored = score(df, bundle)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    print(f"Scored {len(scored)} cycles.")


if __name__ == "__main__":
    main()
