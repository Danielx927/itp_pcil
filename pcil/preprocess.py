"""
PCIL Preprocessing Pipeline — v3 (source-oriented YAML)
=======================================================
Reads the source-oriented `<machine>.yaml` (mqtt / acoustic /
machine_params blocks with explicit `data_type`) and produces the
Golden DataFrame:

    scenario | timestamp | <targets...> | <factors...>     (factors all 0–1)

Run from the PCIL/ root:
    python pcil/preprocess.py                 # default: inkjet_printer
    python pcil/preprocess.py oil_filler      # by machine name
    python pcil/preprocess.py path/to/x.yaml  # by explicit YAML path
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# UTF-8 stdout for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


# ═══════════════════════════════════════════════════════════════
# 0. CLI ARG → CONFIG PATH
# ═══════════════════════════════════════════════════════════════

def resolve_config_path(arg: str | None = None) -> Path:
    """
    Resolve a CLI arg into a YAML path.

    - If arg is a valid file path → use it directly.
    - If arg is a machine name (e.g. "inkjet_printer") →
      machines/<name>/<name>.yaml.
    - If no arg → defaults to inkjet_printer.
    """
    repo_root = Path(__file__).resolve().parent.parent   # PCIL/
    if arg:
        p = Path(arg)
        if p.is_file():
            return p.resolve()
        return repo_root / "machines" / arg / f"{arg}.yaml"
    return repo_root / "machines" / "inkjet_printer" / "inkjet_printer.yaml"


# ═══════════════════════════════════════════════════════════════
# 1. CONFIG LOADING
# ═══════════════════════════════════════════════════════════════

def load_config(config_path: Path) -> dict:
    """Load YAML and resolve all paths against the config file's directory."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    config_dir = config_path.parent
    base_data = (config_dir / cfg["pipeline"]["base_data_dir"]).resolve()
    output_dir = (config_dir / cfg["pipeline"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg["_paths"] = {
        "config_dir":   config_dir,
        "base_data":    base_data,
        "output":       output_dir,
        "mqtt_dir":     base_data / cfg["mqtt"]["data_dir"],
        "acoustic_dir": base_data / cfg["acoustic"]["data_dir"],
        "params_file":  base_data / cfg["machine_params"]["file"],
    }
    return cfg


# ═══════════════════════════════════════════════════════════════
# 2. TYPE-DISPATCHED PREPROCESSING
# ═══════════════════════════════════════════════════════════════

def process_binary_column(series: pd.Series, window: str, aggregation: str) -> pd.Series:
    """0/1 flag → ratio | present | transitions | active_ms."""
    series = series.fillna(0).astype(int)
    if aggregation == "ratio":
        return series.resample(window).mean()
    if aggregation == "present":
        return (series.resample(window).max() > 0).astype(int)
    if aggregation == "transitions":
        return series.diff().abs().resample(window).sum()
    if aggregation == "active_ms":
        return series.resample(window).sum() * 1000
    raise ValueError(f"Unknown binary aggregation: {aggregation}")


def process_numeric_column(series: pd.Series, window: str, aggregation: str) -> pd.Series:
    """Continuous numeric → mean | std | min | max | peak | rms | range | kurtosis | crest."""
    series = series.fillna(0).astype(float)
    r = series.resample(window)
    if aggregation == "mean":     return r.mean()
    if aggregation == "std":      return r.std().fillna(0)
    if aggregation == "min":      return r.min()
    if aggregation == "max":      return r.max()
    if aggregation == "peak":     return r.apply(lambda x: np.abs(x).max() if len(x) else 0)
    if aggregation == "rms":      return r.apply(lambda x: np.sqrt(np.mean(x**2)) if len(x) else 0)
    if aggregation == "range":    return r.max() - r.min()
    if aggregation == "kurtosis": return r.apply(lambda x: pd.Series(x).kurtosis() if len(x) > 3 else 0).fillna(0)
    if aggregation == "crest":
        peak = r.apply(lambda x: np.abs(x).max() if len(x) else 0)
        rms  = r.apply(lambda x: np.sqrt(np.mean(x**2)) if len(x) else 0)
        return (peak / rms.replace(0, np.nan)).fillna(0)
    raise ValueError(f"Unknown numeric aggregation: {aggregation}")


def process_numeric_magnitude_rms(df: pd.DataFrame, cols: list[str], window: str) -> pd.Series:
    """Collapse 3-axis acceleration into one vibration scalar (RMS of magnitude)."""
    df = df[cols].fillna(0).astype(float)
    magnitude = np.sqrt((df ** 2).sum(axis=1))
    return magnitude.resample(window).apply(
        lambda x: np.sqrt(np.mean(x**2)) if len(x) else 0
    )


def process_cyclical_column(series: pd.Series, window: str, aggregation: str) -> pd.Series:
    """
    Cyclical signal handlers (SetVelo / SetPressure).
      cycle_count    → zero-crossings / 2 per window
      cycle_period_* → STUB (falls back to numeric stats)
      dominant_freq  → STUB (falls back to mean)
    Otherwise delegates to process_numeric_column.
    """
    series = series.fillna(0).astype(float)

    if aggregation == "cycle_count":
        detrended = series - series.expanding(min_periods=1).mean()
        crossings = ((detrended.shift(1) * detrended) < 0).astype(int)
        return crossings.resample(window).sum() / 2

    if aggregation in ("cycle_period_mean", "cycle_period_std"):
        return process_numeric_column(series, window, "std")

    if aggregation == "dominant_freq":
        return process_numeric_column(series, window, "mean")

    return process_numeric_column(series, window, aggregation)


def normalize_minmax(series: pd.Series) -> pd.Series:
    """Scale a series to [0, 1]. Flat series → zeros."""
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


# ═══════════════════════════════════════════════════════════════
# 3. OEE TARGETS
# ═══════════════════════════════════════════════════════════════

def compute_oee_targets(oee_cfg: dict) -> dict:
    """availability × performance × quality = oee, per scenario."""
    planned     = oee_cfg["planned_production_time"]
    ideal_cycle = oee_cfg["ideal_cycle_time"]
    out = {}
    for name, s in oee_cfg["scenarios"].items():
        run_time, op_time, good = s["run_time"], s["operating_time"], s["good_units"]
        availability = run_time / planned
        performance  = op_time / run_time if run_time > 0 else 0
        total_units  = op_time / ideal_cycle
        quality      = good / total_units if total_units > 0 else 0
        out[name] = {
            "availability": round(availability, 4),
            "performance":  round(performance,  4),
            "quality":      round(quality,      4),
            "oee":          round(availability * performance * quality, 4),
        }
    return out


# ═══════════════════════════════════════════════════════════════
# 4. LOADERS
# ═══════════════════════════════════════════════════════════════

def load_mqtt_scenario(csv_path: Path, time_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    return df.dropna(subset=[time_col]).set_index(time_col).sort_index()


def load_acoustic_file(csv_path: Path, header_rows: int, sample_rate: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path, skiprows=header_rows)
    n = len(df)
    dt_ns = int(1e9 / sample_rate)
    df.index = pd.to_datetime(np.arange(n) * dt_ns, unit="ns")
    df.index.name = "timestamp"
    return df


def load_machine_params(csv_path: Path, time_col: str, sep: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=sep)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    return df.dropna(subset=[time_col]).set_index(time_col).sort_index()


# ═══════════════════════════════════════════════════════════════
# 5. FACTOR DISPATCH (schema.factors → Series)
# ═══════════════════════════════════════════════════════════════

def compute_mqtt_factor(mqtt_df: pd.DataFrame, factor: dict, window: str) -> pd.Series:
    col = factor["column"]
    if col not in mqtt_df.columns:
        return pd.Series(dtype=float)
    if factor["type"] == "binary":
        return process_binary_column(mqtt_df[col], window, factor["aggregation"])
    if factor["type"] == "numeric":
        return process_numeric_column(mqtt_df[col], window, factor["aggregation"])
    raise ValueError(f"Unsupported MQTT factor type: {factor['type']}")


def compute_acoustic_factor(acc_df: pd.DataFrame, factor: dict, window: str) -> pd.Series:
    agg = factor["aggregation"]
    if agg == "magnitude_rms":
        return process_numeric_magnitude_rms(acc_df, factor["columns"], window)
    if "column" in factor:
        return process_numeric_column(acc_df[factor["column"]], window, agg)
    raise ValueError(f"Unsupported acoustic factor: {factor['name']}")


def compute_param_factor(param_df: pd.DataFrame, factor: dict, window: str) -> pd.Series:
    col = factor["column"]
    if col not in param_df.columns:
        return pd.Series(dtype=float)
    if factor["type"] == "numeric_cyclical":
        return process_cyclical_column(param_df[col], window, factor["aggregation"])
    if factor["type"] == "numeric":
        return process_numeric_column(param_df[col], window, factor["aggregation"])
    raise ValueError(f"Unsupported machine_params factor type: {factor['type']}")


# ═══════════════════════════════════════════════════════════════
# 6. BUILD ONE SCENARIO'S ROWS
# ═══════════════════════════════════════════════════════════════

def _align_to_timeline(source: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    n = len(target_index)
    vals = source.values
    if len(vals) == 0:
        return pd.Series(np.zeros(n), index=target_index)
    if len(vals) >= n:
        return pd.Series(vals[:n], index=target_index)
    reps = int(np.ceil(n / len(vals)))
    return pd.Series(np.tile(vals, reps)[:n], index=target_index)


def build_scenario_rows(
    scenario_name: str,
    scenario_cfg: dict,
    cfg: dict,
    targets_for_scenario: dict,
    acoustic_cache: dict,
    param_df: pd.DataFrame,
) -> pd.DataFrame:
    window = cfg["pipeline"]["window_size"]

    mqtt_df = load_mqtt_scenario(
        cfg["_paths"]["mqtt_dir"] / scenario_cfg["file"],
        cfg["mqtt"]["time_column"],
    )

    factors = cfg["schema"]["factors"]
    series_by_name: dict[str, pd.Series] = {}

    for f in (x for x in factors if x["source"] == "mqtt"):
        series_by_name[f["name"]] = compute_mqtt_factor(mqtt_df, f, window)

    if not series_by_name:
        raise RuntimeError(
            f"No MQTT factors for scenario '{scenario_name}' — "
            f"schema.factors must include at least one mqtt factor"
        )
    timeline = next(iter(series_by_name.values())).index

    acoustic_key = scenario_cfg.get("acoustic_mapping")
    if acoustic_key and acoustic_key in acoustic_cache:
        acc_df = acoustic_cache[acoustic_key]
        for f in (x for x in factors if x["source"] == "acoustic"):
            s = compute_acoustic_factor(acc_df, f, window)
            series_by_name[f["name"]] = _align_to_timeline(s, timeline)

    for f in (x for x in factors if x["source"] == "machine_params"):
        s = compute_param_factor(param_df, f, window)
        series_by_name[f["name"]] = _align_to_timeline(s, timeline)

    out = pd.DataFrame(index=timeline)
    out.index.name = cfg["schema"]["timestamp_column"]
    out.insert(0, cfg["schema"]["scenario_column"], scenario_name)

    for t in cfg["schema"]["targets"]:
        out[t["name"]] = targets_for_scenario[t["name"]]

    for f in factors:
        out[f["name"]] = series_by_name.get(f["name"], pd.Series(dtype=float))\
            .reindex(timeline).fillna(0)

    return out.reset_index()


# ═══════════════════════════════════════════════════════════════
# 7. BUILD THE GOLDEN DATAFRAME
# ═══════════════════════════════════════════════════════════════

def build_golden_dataframe(cfg: dict) -> pd.DataFrame:
    targets_per_scenario = compute_oee_targets(cfg["oee"])

    acoustic_cache = {}
    for cond_name, cond in cfg["acoustic"]["conditions"].items():
        path = cfg["_paths"]["acoustic_dir"] / cond["file"]
        acoustic_cache[cond_name] = load_acoustic_file(
            path, cfg["acoustic"]["header_rows"], cfg["acoustic"]["sample_rate"]
        )

    param_df = load_machine_params(
        cfg["_paths"]["params_file"],
        cfg["machine_params"]["time_column"],
        cfg["machine_params"]["separator"],
    )

    frames = []
    for name, s_cfg in tqdm(cfg["mqtt"]["scenarios"].items(), desc="Scenarios"):
        frames.append(build_scenario_rows(
            name, s_cfg, cfg, targets_per_scenario[name], acoustic_cache, param_df,
        ))

    golden = pd.concat(frames, ignore_index=True)

    for f in cfg["schema"]["factors"]:
        if f.get("normalize") == "minmax":
            golden[f["name"]] = normalize_minmax(golden[f["name"]])
    for t in cfg["schema"]["targets"]:
        if t.get("normalize") == "minmax":
            golden[t["name"]] = normalize_minmax(golden[t["name"]])

    ordered = (
        [cfg["schema"]["scenario_column"], cfg["schema"]["timestamp_column"]]
        + [t["name"] for t in cfg["schema"]["targets"]]
        + [f["name"] for f in cfg["schema"]["factors"]]
    )
    return golden[ordered]


# ═══════════════════════════════════════════════════════════════
# 8. REPORT
# ═══════════════════════════════════════════════════════════════

def print_summary(golden: pd.DataFrame, cfg: dict) -> None:
    factor_names = [f["name"] for f in cfg["schema"]["factors"]]
    target_names = [t["name"] for t in cfg["schema"]["targets"]]
    print("\n" + "=" * 70)
    print(f"GOLDEN DATAFRAME v3 — {cfg['machine']['name']}")
    print("=" * 70)
    print(f"Shape: {golden.shape[0]} rows x {golden.shape[1]} cols")
    print(f"Columns: {list(golden.columns)}")
    print()
    print("Rows per scenario:")
    print(golden[cfg["schema"]["scenario_column"]].value_counts().to_string())
    print()
    print("Factor value ranges (factors flagged minmax should sit in [0, 1]):")
    print(golden[factor_names].agg(["min", "max", "mean"]).round(4).to_string())
    print()
    print("OEE targets per scenario:")
    per_scen = golden.groupby(cfg["schema"]["scenario_column"])[target_names].first()
    print(per_scen.round(4).to_string())
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════
# 9. MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    config_path = resolve_config_path(arg)
    if not config_path.is_file():
        print(f"Config not found: {config_path}")
        raise SystemExit(1)

    print(f"[1/3] Loading config: {config_path}")
    cfg = load_config(config_path)

    print(f"[2/3] Building Golden DataFrame...")
    golden = build_golden_dataframe(cfg)

    out_path = cfg["_paths"]["output"] / "golden_dataframe.csv"
    golden.to_csv(out_path, index=False)
    print(f"[3/3] Saved -> {out_path}")

    print_summary(golden, cfg)


if __name__ == "__main__":
    main()
