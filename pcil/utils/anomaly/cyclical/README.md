# Cyclical anomaly pipeline

Picks up where the menu's task 1 leaves off. This subpackage is the
**pipeline definition** for cyclical data — the engineering team will
run it per-machine to produce per-machine `.pkl` files.

## Status

| File | What it does | Status |
|---|---|---|
| `prepare_data.py` | `Clean_Data.csv` -> `cyclical_dataset.csv` + `cyclical_eval.csv` | **TODO** — function bodies stubbed; CLI wiring done |
| `slice.py` | cycle detection | **TODO** — pick peak / zero-crossing / fixed-period and implement |
| `features.py` | per-cycle feature vector | **TODO** — pick stats / resampled / FFT and implement |
| `model.py` | candidate models (z-score, IF, OCSVM, autoencoder) | **TODO** — pick one, implement; the others can stay stubbed |
| `train.py` | CLI: dataset -> trained `.pkl` | done — wiring complete; runs end-to-end once the above are filled in |
| `score.py` | CLI: slice + `.pkl` -> `anomaly_score` per cycle | done — wiring complete |

## Recommended order to fill in

1. `prepare_data.py` — produces the input files for `train.py`.
2. `slice.py` — pick a cycle detection method.
3. `features.py` — pick a feature set.
4. `model.py` — pick **one** candidate, implement, leave the others as
   stubs (or implement extras to compare).
5. Run `train.py` end-to-end on `cyclical_dataset.csv`.
6. Run `score.py` against `cyclical_eval.csv`, compare predictions to
   the `cycle_label` ground truth, report precision / recall.

## Heads up about the data

- `data/Clean_Data.csv` is logged at 1 kHz but the source samples at
  ~250 Hz, so values appear in 4-row groups. Decimate to 250 Hz or
  smooth before peak detection.
- `SetPressure` may be a commanded setpoint, not a measured sensor —
  Dion has a question out to Winardi.

## Spec each design choice in this README before merging

- **Slicing method:** _pick one and write 1–2 lines on why._
- **Feature set:** _pick one and write 1–2 lines on why._
- **Model:** _pick one. If autoencoder: layers + bottleneck + loss + optimiser + stopping criterion._
