"""
Run visual checks for every cyclical slicing method.

This is a lightweight inspection pipeline: it reads `cyclical_eval.csv`,
applies each slicing method, and writes one 3x2 normal/anomalous
comparison plot per method.

Run from the repo root:
    python -m pcil.utils.anomaly.cyclical.test_slicing_pipeline \
        --input data/cyclical/cyclical_eval.csv \
        --output-dir data/cyclical/slicing_visuals
"""

from __future__ import annotations

import argparse

import pandas as pd

from pcil.utils.anomaly.cyclical.visualization import plot_all_slicing_methods


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 3x2 plots for all slicing methods.")
    parser.add_argument("--input", required=True, help="Path to cyclical_eval.csv.")
    parser.add_argument("--output-dir", required=True, help="Directory to write PNG plots.")
    parser.add_argument("--machine-id", default=None, help="Optional machine_id to plot.")
    parser.add_argument("--examples-per-label", type=int, default=3)
    parser.add_argument("--window-size-rows", type=int, default=1000)
    parser.add_argument("--signal-column", default="signal_value")
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--machine-id-column", default="machine_id")
    parser.add_argument("--label-column", default="cycle_label")
    args = parser.parse_args()

    eval_df = pd.read_csv(args.input)
    outputs = plot_all_slicing_methods(
        eval_df,
        output_dir=args.output_dir,
        machine_id=args.machine_id,
        examples_per_label=args.examples_per_label,
        window_size_rows=args.window_size_rows,
        signal_column=args.signal_column,
        timestamp_column=args.timestamp_column,
        machine_id_column=args.machine_id_column,
        label_column=args.label_column,
    )
    for output in outputs:
        print(f"Saved -> {output}")


if __name__ == "__main__":
    main()
