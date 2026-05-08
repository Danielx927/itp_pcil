"""
Pipeline #2 v0 — Context Model (Linear Regression)
==================================================
Reads the Golden DataFrame produced by preprocess.py, pushes it
through the adapter, fits a multi-target linear regression, and
writes:
  - context_model_impacts.json  (Pipeline #2 output schema)
  - context_model.pkl            (the trained model, ready to reuse)

Run from PCIL_dev/:
    python pcil/train_context_model.py                # default: inkjet_printer
    python pcil/train_context_model.py oil_filler     # by machine name
"""

from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression

from adapter import adapt, column_names_from_config

# UTF-8 stdout for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def _resolve_machine(arg: str | None) -> Path:
    repo_root = Path(__file__).resolve().parent.parent  # PCIL_dev/
    if arg:
        p = Path(arg)
        if p.is_file():
            return p.resolve()
        return repo_root / "machines" / arg / "config.yaml"
    return repo_root / "machines" / "inkjet_printer" / "config.yaml"


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    cfg_path = _resolve_machine(arg)
    if not cfg_path.is_file():
        print(f"Config not found: {cfg_path}")
        raise SystemExit(1)

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    machine_dir = cfg_path.parent
    output_dir = (machine_dir / cfg["pipeline"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "golden_dataframe.csv"
    if not csv_path.exists():
        print(f"Golden DataFrame not found at {csv_path}")
        print(f"Run preprocess.py first.")
        raise SystemExit(1)

    df = pd.read_csv(csv_path)

    # 1. Adapt — derive features from the Golden DataFrame columns so we
    # capture any post-OneHotEncoder column expansion correctly.
    targets, features = column_names_from_config(cfg, df)
    bundle = adapt(df, targets, features)
    X, y = bundle["X"], bundle["y"]

    # 2. Fit multi-target linear regression
    model = LinearRegression().fit(X, y)

    # 3. Build per-target impact blocks
    timestamp_col = cfg["input"]["timestamp_column"]
    timestamps = pd.to_datetime(df[timestamp_col])
    time_from = timestamps.min().isoformat()
    time_to   = timestamps.max().isoformat()

    blocks = []
    for i, target_name in enumerate(targets):
        blocks.append({
            "time_from": time_from,
            "time_to":   time_to,
            "target":    target_name,
            "intercept": float(model.intercept_[i]),
            "feature_impacts": {
                feat: float(coef)
                for feat, coef in zip(features, model.coef_[i])
            },
        })

    out = {
        "model":      "linear_regression",
        "machine":    machine_dir.name,
        "fitted_at":  datetime.now(timezone.utc).isoformat(),
        "n_rows":     bundle["n_rows"],
        "n_features": len(features),
        "n_targets":  len(targets),
        "blocks":     blocks,
    }

    # 4. Save artefacts
    json_path = output_dir / "context_model_impacts.json"
    pkl_path  = output_dir / "context_model.pkl"
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    joblib.dump({
        "model":         model,
        "feature_names": features,
        "target_names":  targets,
    }, pkl_path)

    print(f"Wrote {json_path}")
    print(f"Wrote {pkl_path}")
    print()
    print("Linear-regression feature impacts (per target):")
    print(f"  features: {features}")
    print(f"  targets:  {targets}")
    print()
    for block in blocks:
        print(f"  Target: {block['target']}   (intercept {block['intercept']:+.4f})")
        for feat, coef in block["feature_impacts"].items():
            print(f"    {feat:<32s} {coef:+.4f}")
        print()


if __name__ == "__main__":
    main()
