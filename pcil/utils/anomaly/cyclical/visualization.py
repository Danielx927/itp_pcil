"""
Cyclical anomaly pipeline — visualization helper
================================================
Plot representative normal and anomalous cycles from `cyclical_eval.csv`.

Run from the repo root:
    python -m pcil.utils.anomaly.cyclical.visualization \
        --input data/cyclical/cyclical_eval.csv \
        --output data/cyclical/cycle_comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pcil.utils.anomaly.cyclical.slice import detect_cycles


SLICING_METHODS = ("peak", "zero-crossing", "fixed-period")


def plot_cycle_comparison(
    eval_df: pd.DataFrame,
    *,
    output: str | Path,
    machine_id: str | None = None,
    examples_per_label: int = 3,
    slicing_method: str = "fixed-period",
    window_size_rows: int = 1000,
    signal_column: str = "signal_value",
    timestamp_column: str = "timestamp",
    machine_id_column: str = "machine_id",
    label_column: str = "cycle_label",
) -> Path:
    """
    Save a side-by-side plot of normal and anomalous cycle examples.

    Cycle boundaries come from `slice.detect_cycles`, and each sliced
    cycle is labelled anomalous if any row inside it has an anomalous
    `cycle_label`.
    """
    required = {signal_column, label_column}
    missing = required - set(eval_df.columns)
    if missing:
        raise KeyError(f"Missing required column(s): {sorted(missing)}")
    if window_size_rows <= 0:
        raise ValueError("window_size_rows must be positive")
    if examples_per_label <= 0:
        raise ValueError("examples_per_label must be positive")

    df = eval_df.copy()
    if machine_id is not None:
        if machine_id_column not in df.columns:
            raise KeyError(f"Missing machine id column: {machine_id_column}")
        df = df[df[machine_id_column] == machine_id].copy()
        if df.empty:
            raise ValueError(f"No rows found for machine_id={machine_id!r}")
    elif machine_id_column in df.columns:
        machine_id = str(df[machine_id_column].iloc[0])
        df = df[df[machine_id_column] == machine_id].copy()

    if timestamp_column in df.columns:
        df = df.sort_values(timestamp_column).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    windows = _summarise_cycles(
        df,
        slicing_method=slicing_method,
        window_size_rows=window_size_rows,
        signal_column=signal_column,
        timestamp_column=timestamp_column,
        label_column=label_column,
    )
    normal_windows = [w for w in windows if w["label"] == "normal"][:examples_per_label]
    anomalous_windows = [w for w in windows if w["label"] == "anomalous"][:examples_per_label]

    if not normal_windows or not anomalous_windows:
        counts = df[label_column].value_counts().to_dict()
        raise ValueError(
            "Need at least one normal and one anomalous window to plot. "
            f"Observed row label counts: {counts}. "
            f"Slicing method {slicing_method!r} produced {len(windows)} cycles."
        )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    row_count = max(len(normal_windows), len(anomalous_windows))
    fig, axes = plt.subplots(row_count, 2, figsize=(14, 3.2 * row_count), sharex=False)
    if row_count == 1:
        axes = np.array([axes])
    fig.suptitle(
        f"Cyclical eval comparison ({slicing_method})"
        f"{f' — {machine_id}' if machine_id else ''}"
    )

    _plot_window_column(
        axes[:, 0],
        df,
        normal_windows,
        signal_column=signal_column,
        label="Normal",
        color="#2563eb",
    )
    _plot_window_column(
        axes[:, 1],
        df,
        anomalous_windows,
        signal_column=signal_column,
        label="Anomalous",
        color="#dc2626",
    )

    for ax in axes[-1, :]:
        ax.set_xlabel("Sample within window")
    for ax in axes.ravel():
        ax.set_ylabel(signal_column)
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_all_slicing_methods(
    eval_df: pd.DataFrame,
    *,
    output_dir: str | Path,
    output_prefix: str = "cycle_comparison",
    examples_per_label: int = 3,
    **kwargs,
) -> list[Path]:
    """Generate one comparison plot per supported slicing method."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for method in SLICING_METHODS:
        slug = method.replace("-", "_")
        output = output_dir / f"{output_prefix}_{slug}_3x2.png"
        outputs.append(
            plot_cycle_comparison(
                eval_df,
                output=output,
                examples_per_label=examples_per_label,
                slicing_method=method,
                **kwargs,
            )
        )
    return outputs


def _summarise_cycles(
    df: pd.DataFrame,
    *,
    slicing_method: str,
    window_size_rows: int,
    signal_column: str,
    timestamp_column: str,
    label_column: str,
) -> list[dict[str, int | str]]:
    cycles = []
    for cycle_id, (start, end) in enumerate(
        detect_cycles(
            df,
            signal_column=signal_column,
            timestamp_column=timestamp_column,
            method=slicing_method,
            fixed_period_rows=window_size_rows,
        )
    ):
        labels = set(df.iloc[start:end][label_column].astype(str))
        label = "anomalous" if "anomalous" in labels else "normal"
        cycles.append(
            {
                "id": cycle_id,
                "start": int(start),
                "end": int(end),
                "label": label,
            }
        )
    return cycles


def _window_signal(
    df: pd.DataFrame,
    window: dict[str, int | str],
    *,
    signal_column: str,
) -> np.ndarray:
    start = int(window["start"])
    end = int(window["end"])
    return df.iloc[start:end][signal_column].to_numpy(dtype=float)


def _plot_single_cycle(
    ax,
    signal: np.ndarray,
    *,
    title: str,
    color: str,
) -> None:
    x = np.arange(len(signal))
    ax.plot(x, signal, color=color, linewidth=1.25)
    ax.set_title(title)


def _plot_window_column(
    axes,
    df: pd.DataFrame,
    windows: list[dict[str, int | str]],
    *,
    signal_column: str,
    label: str,
    color: str,
) -> None:
    for i, ax in enumerate(axes):
        if i >= len(windows):
            ax.axis("off")
            continue

        window = windows[i]
        signal = _window_signal(df, window, signal_column=signal_column)
        _plot_single_cycle(
            ax,
            signal,
            title=f"{label} window {window['id']} ({window['start']}:{window['end']})",
            color=color,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot normal vs anomalous cyclical eval windows.")
    parser.add_argument("--input", required=True, help="Path to cyclical_eval.csv.")
    parser.add_argument("--output", required=True, help="Path to write the PNG plot.")
    parser.add_argument(
        "--all-methods",
        action="store_true",
        help="Generate one 3x2 plot for peak, zero-crossing, and fixed-period.",
    )
    parser.add_argument("--machine-id", default=None, help="Optional machine_id to plot.")
    parser.add_argument("--examples-per-label", type=int, default=3)
    parser.add_argument("--slicing-method", choices=SLICING_METHODS, default="fixed-period")
    parser.add_argument("--window-size-rows", type=int, default=1000)
    parser.add_argument("--signal-column", default="signal_value")
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--machine-id-column", default="machine_id")
    parser.add_argument("--label-column", default="cycle_label")
    args = parser.parse_args()

    eval_df = pd.read_csv(args.input)
    if args.all_methods:
        out_paths = plot_all_slicing_methods(
            eval_df,
            output_dir=args.output,
            machine_id=args.machine_id,
            examples_per_label=args.examples_per_label,
            window_size_rows=args.window_size_rows,
            signal_column=args.signal_column,
            timestamp_column=args.timestamp_column,
            machine_id_column=args.machine_id_column,
            label_column=args.label_column,
        )
        for out_path in out_paths:
            print(f"Saved -> {out_path}")
    else:
        out_path = plot_cycle_comparison(
            eval_df,
            output=args.output,
            machine_id=args.machine_id,
            examples_per_label=args.examples_per_label,
            slicing_method=args.slicing_method,
            window_size_rows=args.window_size_rows,
            signal_column=args.signal_column,
            timestamp_column=args.timestamp_column,
            machine_id_column=args.machine_id_column,
            label_column=args.label_column,
        )
        print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
