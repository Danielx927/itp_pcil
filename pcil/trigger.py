"""
PCIL Trigger — generic time-slicer for tabular machine data
============================================================
Selects which rows of a DataFrame go downstream into preprocessing.
Two slice modes:
  - slice_by_time(df, start, end)   -> rows in [start, end]
  - slice_last_n_rows(df, n)        -> the most recent n rows

When the shop-floor database goes live, this same module will pull
slices from Postgres instead of accepting an in-memory DataFrame; the
function signatures should stay the same so callers don't break.

Run from PCIL_dev/:
    python -m pcil.trigger --input slice.csv --start "2026-05-08 10:00:00" --end "2026-05-08 10:05:00"
    python -m pcil.trigger --input slice.csv --last 100
"""

from __future__ import annotations

import argparse
from typing import Union

import pandas as pd

TimestampLike = Union[str, pd.Timestamp]


def slice_by_time(
    df: pd.DataFrame,
    start_time: TimestampLike,
    end_time: TimestampLike,
    *,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """
    Return rows where df[timestamp_column] is between start_time and end_time (inclusive).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a timestamp column (default name: "timestamp").
    start_time, end_time : str | pd.Timestamp
        ISO 8601 strings or pandas Timestamps. UTC for now.
    timestamp_column : str, default "timestamp"

    Returns
    -------
    pd.DataFrame
        Filtered rows, same columns as input.

    TODO (teammate):
      1. Convert start_time / end_time to pd.Timestamp via pd.to_datetime.
      2. Make sure df[timestamp_column] is datetime dtype (call pd.to_datetime if needed).
      3. Filter and return.
      4. Edge cases to consider: start > end (return empty?), missing column (raise).
    """
    raise NotImplementedError("TODO: implement slice_by_time")


def slice_last_n_rows(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Return the most recent n rows of df.

    Parameters
    ----------
    df : pd.DataFrame
    n : int
        Number of rows. If n >= len(df), return the whole frame.
        If n <= 0, return an empty frame with the same columns.

    Returns
    -------
    pd.DataFrame

    TODO (teammate):
      1. Handle n <= 0 (return df.iloc[0:0]).
      2. Use df.tail(n) — make sure the index is reset if needed.
    """
    raise NotImplementedError("TODO: implement slice_last_n_rows")


# ─────────────────────────────────────────────────────────────
# CLI demo
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PCIL trigger — slice rows from a CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m pcil.trigger --input slice.csv --start '2026-05-08 10:00:00' --end '2026-05-08 10:05:00'\n"
            "  python -m pcil.trigger --input slice.csv --last 100\n"
        ),
    )
    parser.add_argument("--input", required=True, help="CSV with a timestamp column.")
    parser.add_argument("--start", default=None, help="ISO 8601 start time")
    parser.add_argument("--end", default=None, help="ISO 8601 end time")
    parser.add_argument("--last", type=int, default=None, help="Return the last N rows")
    parser.add_argument("--timestamp-column", default="timestamp")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    if args.start and args.end:
        out = slice_by_time(df, args.start, args.end, timestamp_column=args.timestamp_column)
    elif args.last is not None:
        out = slice_last_n_rows(df, args.last)
    else:
        raise SystemExit("Pass either --start/--end or --last.")

    print(out.head(10).to_string())
    print(f"\nReturned {len(out)} rows.")


if __name__ == "__main__":
    main()
