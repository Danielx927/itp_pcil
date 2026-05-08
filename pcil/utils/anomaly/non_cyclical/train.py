"""
Non-cyclical anomaly pipeline — train CLI
==========================================
Wires the four steps together:
  1. window slicing                     -> non_cyclical/slice.py
  2. extract features per window        -> non_cyclical/features.py
  3. per-machine normalisation          -> shared normalise.PerMachineNormaliser
  4. fit the chosen model               -> non_cyclical/model.py

Acoustic CSVs have 5 metadata header rows; we skip them on load.
Trains only on the *_clean.csv file(s) — anomaly files are reserved for
score-time evaluation.

Run from PCIL_dev/:
    python -m pcil.utils.anomaly.non_cyclical.train \\
        --input "../data/Inkjet Printer Data Collection/Acoustic Sensor Data/machine_on_clean.csv" \\
        --output ../data/non_cyclical_inkjet_01.pkl \\
        --machine-id inkjet_01 \\
        --window-size-rows 12800 \\
        --model isolation_forest

Models: z_score | isolation_forest | one_class_svm | autoencoder

The reusable deliverable is this pipeline code. The saved .pkl is a
fitted instance for the machine/data type used during training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from pcil.utils.anomaly.normalise import PerMachineNormaliser
from pcil.utils.anomaly.non_cyclical.slice import detect_windows
from pcil.utils.anomaly.non_cyclical.features import (
    CHANNEL_COLUMNS,
    extract_features,
    stack_features,
)
from pcil.utils.anomaly.non_cyclical.model import (
    AutoencoderModel,
    IsolationForestModel,
    OneClassSVMModel,
    ZScoreModel,
)

_MODEL_REGISTRY = {
    "z_score":          ZScoreModel,
    "isolation_forest": IsolationForestModel,
    "one_class_svm":    OneClassSVMModel,
    "autoencoder":      AutoencoderModel,
}


def load_acoustic(csv_path: Path, *, header_skiprows: int = 5) -> pd.DataFrame:
    """Load an acoustic CSV, skipping its 5 metadata lines."""
    return pd.read_csv(csv_path, skiprows=header_skiprows)


def train(
    df: pd.DataFrame,
    model_name: str,
    *,
    machine_id: str,
    window_size_rows: int,
    channel_columns: list[str] = CHANNEL_COLUMNS,
    model_kwargs: dict | None = None,
) -> dict:
    """
    Run the full pipeline on a single-machine clean DataFrame.

    Single-machine scope is intentional: run the same pipeline definition
    once per machine to produce that machine's fitted normaliser/model
    bundle.
    """
    # 1+2. Window the data, extract features per window
    rows = []
    df_sorted = df.reset_index(drop=True)
    for start, end in detect_windows(df_sorted, window_size_rows=window_size_rows):
        features = extract_features(df_sorted.iloc[start:end], channel_columns=channel_columns)
        features["machine_id"] = machine_id
        rows.append(features)

    feature_df = stack_features(rows)
    feature_columns = [c for c in feature_df.columns if c != "machine_id"]

    # 3. Per-machine normalisation
    normaliser = PerMachineNormaliser()
    feature_df_norm = normaliser.fit_transform(
        feature_df,
        machine_id_column="machine_id",
        feature_columns=feature_columns,
    )

    # 4. Fit the chosen model
    model_cls = _MODEL_REGISTRY[model_name]
    model = model_cls(**(model_kwargs or {}))
    X = feature_df_norm[feature_columns].to_numpy(dtype=float)
    model.fit(X)

    return {
        "model":             model,
        "model_name":        model_name,
        "normaliser":        normaliser,
        "feature_columns":   feature_columns,
        "machine_id":        machine_id,
        "trained_machine_ids": [machine_id],
        "machine_id_column": "machine_id",
        "window_size_rows":  window_size_rows,
        "channel_columns":   list(channel_columns),
    }


def main():
    parser = argparse.ArgumentParser(description="Train non-cyclical anomaly model.")
    parser.add_argument("--input", required=True, help="Path to *_clean.csv (acoustic).")
    parser.add_argument("--output", required=True, help="Where to save the bundled .pkl.")
    parser.add_argument("--model", choices=list(_MODEL_REGISTRY), default="isolation_forest")
    parser.add_argument("--machine-id", default="inkjet_01")
    parser.add_argument("--window-size-rows", type=int, required=True,
                        help="Window length in rows. 12800 = 0.5s at 25.6 kHz.")
    parser.add_argument("--header-skiprows", type=int, default=5,
                        help="Metadata lines to skip in the CSV.")
    args = parser.parse_args()

    df = load_acoustic(Path(args.input), header_skiprows=args.header_skiprows)
    bundle = train(
        df,
        model_name=args.model,
        machine_id=args.machine_id,
        window_size_rows=args.window_size_rows,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    print(f"Saved -> {out_path}")
    print(f"Trained {args.model} on {len(bundle['feature_columns'])} features.")


if __name__ == "__main__":
    main()
