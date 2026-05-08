# Non-cyclical anomaly pipeline

Picks up where the menu's task 2 leaves off. This subpackage is the
**pipeline definition** for non-cyclical (continuous, no clear repeat)
data — vibration / acoustic / temperature streams.

## Status

| File | What it does | Status |
|---|---|---|
| `slice.py` | fixed-window slicing | **TODO** — pick window length, implement |
| `features.py` | per-window feature vector (RMS / FFT / kurtosis / ...) | **TODO** — pick feature set, implement |
| `model.py` | candidate models | **TODO** — pick one, implement |
| `train.py` | CLI: clean files -> trained `.pkl` | done — wiring complete |
| `score.py` | CLI: data + `.pkl` -> `anomaly_score` | done — wiring complete |

## Free ground truth

The acoustic dataset already has clean / anomaly variants per filename.
No synthetic data prep needed:

```
data/Inkjet Printer Data Collection/Acoustic Sensor Data/
├── machine_on_clean.csv     (train + eval-normal)
├── machine_on_anomaly.csv   (eval-anomalous)
├── machine_off_clean.csv
└── machine_off_anomaly.csv
```

Train on `machine_on_clean.csv` windows, evaluate on
`machine_on_anomaly.csv` windows. Use filenames as ground-truth labels.

## Heads up about the data

- **Skip 5 metadata header rows** when loading
  (`pd.read_csv(..., skiprows=5)` or similar).
- 25.6 kHz sample rate. With 4 channels (Acceleration 0/1/2 + AE),
  files can be sizeable — load in chunks if needed, or downsample.
- Only ONE machine is available in the dataset. That is fine for the
  intended workflow: train one fitted bundle per machine. Note in the
  writeup that this dataset cannot test a second real machine's separate
  baseline/model instance.

## Recommended order to fill in

1. `slice.py` — pick window length (0.1 s? 0.5 s?) with reasoning.
2. `features.py` — pick feature set per channel (RMS, FFT bands, ...).
3. `model.py` — pick one candidate.
4. Run `train.py` against `machine_on_clean.csv`.
5. Run `score.py` against `machine_on_anomaly.csv`; compare predictions
   to the file label, report precision / recall.

## Spec each design choice in this README before merging

- **Window length:** _pick one + rationale._
- **Features per channel:** _pick set + rationale._
- **Model:** _pick one. If autoencoder: full architecture spec._
