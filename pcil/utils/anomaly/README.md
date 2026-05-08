# pcil/utils/anomaly/

Anomaly detection utilities. The engineering team imports these into
their ingestion path; the resulting `anomaly_score` is appended as a
column to the shop-floor database.

## Pipeline-vs-instance framing (Winardi, 2026-05-08)

We deliver **the pipeline definition**, not one trained model instance
forced to work across all machines. Each machine runs the same pipeline
code on its own data and gets its own fitted `.pkl` bundle. Same recipe,
different fitted normaliser statistics and model weights per machine.
At scale:

```
N machines x M data types = N*M trained .pkl files in production
                            (we ship the M pipelines)
```

## Data types

| Subpackage | Slicing | Example data |
|---|---|---|
| `cyclical/` | cycle detection (peak / zero-crossing / fixed period) | `data/Clean_Data.csv` (SetPressure) |
| `non_cyclical/` | fixed time windows | `data/Inkjet Printer Data Collection/Acoustic Sensor Data/` |

## Four-step structure (per data type)

1. **Slicing** — `slice.py`: cut the raw stream into discrete units.
2. **Feature extraction** — `features.py`: per-unit feature vector.
3. **Per-machine normalisation** — shared `normalise.py`: fit baseline
   mean/std on the machine's own training data, then reuse that fitted
   normaliser when scoring the same machine.
4. **Model architecture + training** — `model.py` + `train.py`:
   pick z-score / Isolation Forest / one-class SVM / autoencoder.

## Shared utilities

- **`normalise.PerMachineNormaliser`** — fit on the training/baseline
  data for a machine; transform z-scores each row against the fitted
  mean and std for that same machine. The fitted normaliser is saved
  inside that machine's trained `.pkl` so scoring can re-apply it.

## Runtime contract

The engineering team uses these utilities in two modes:

**Training (offline, once per machine and data type):**
```python
from pcil.utils.anomaly.cyclical.train import train

bundle = train(historical_df, model_name="isolation_forest")
joblib.dump(bundle, "cyclical_inkjet_01.pkl")
```

**Inference (online, streaming, using that same machine's bundle):**
```python
from pcil.utils.anomaly.cyclical.score import score
import joblib

bundle = joblib.load("cyclical_inkjet_01.pkl")
scored = score(new_slice, bundle)  # adds 'anomaly_score' column
```

## Status

- `normalise.py` — implemented.
- `cyclical/` — wiring in place; teammate fills in slice / features / model.
- `non_cyclical/` — wiring in place; teammate fills in slice / features / model.
