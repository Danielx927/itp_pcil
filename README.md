# PCIL — Production Context Intelligence Layer

A pipeline that watches a factory machine, figures out what is slowing it
down, and produces an operator-readable explanation. The first machine is
an inkjet printer at A*STAR SIMTech.

ITP project · Dion Ko (2401112), Zi Hin, Robin, Daniel, Jaymon · Supervisor: Winardi.

---

## What it does

```
shop-floor DB --> trigger slice --> Pipeline #1 --> Golden DataFrame --> Pipeline #2 --> Pipeline #3 --> dashboard
                                    (preprocess)    (clean spreadsheet)  (math model)    (LLM + RAG)     (text)
```

- **Pipeline #1** takes a slice of the shared shop-floor DataFrame
  (provided as a CSV here, will be a Postgres pull later), passes it
  through a `sklearn.pipeline.Pipeline` + `ColumnTransformer`
  (`MinMaxScaler` for numerical features, `OneHotEncoder` for
  categorical), and produces a Golden DataFrame whose feature values
  sit in [0, 1].
- **Pipeline #2** fits a model that says *"feature X had impact +0.4 on
  OEE"*. v0 is linear regression; SHAP comes later.
- **Pipeline #3** is the LLM stage that turns those impacts plus a
  recovery-document RAG into a sentence the operator can act on. Not
  implemented yet — Winardi's reference docs are in `data/RAG/`.

The per-CSV ingestion (MQTT error flags, 25 kHz acoustic, machine
parameters) is no longer Pipeline #1's responsibility — the
engineering team owns that upstream and writes results to the
shop-floor database.

---

## Project structure

```
PCIL/
├── pcil/                            # shared code, used by every machine
│   ├── preprocess.py                # shop-floor slice -> Golden DataFrame (sklearn Pipeline + ColumnTransformer)
│   ├── adapter.py                   # Golden DataFrame -> numpy arrays for ML
│   ├── train_context_model.py       # fits LinearRegression, writes impact JSON + .pkl
│   ├── trigger.py                   # generic time-slicer (slice_by_time / slice_last_n_rows)  [skeleton — task 3]
│   ├── rag/                         # Pipeline #3: recovery-doc retrieval prototype           [skeleton — task 4]
│   │   ├── loader.py                #   DOCX parser
│   │   ├── lookup.py                #   keyword search
│   │   ├── prototype.py             #   end-to-end demo CLI
│   │   └── README.md
│   └── utils/
│       └── anomaly/                 # engineer-facing anomaly utilities                       [skeleton — tasks 1 + 2]
│           ├── normalise.py         #   per-machine z-score helper, fitted inside each anomaly bundle (working)
│           ├── README.md
│           ├── cyclical/            #   cyclical pipeline (Clean_Data.csv)                   [task 1]
│           │   ├── prepare_data.py
│           │   ├── slice.py
│           │   ├── features.py
│           │   ├── model.py
│           │   ├── train.py
│           │   ├── score.py
│           │   └── README.md
│           └── non_cyclical/        #   non-cyclical pipeline (acoustic data)                [task 2]
│               ├── slice.py
│               ├── features.py
│               ├── model.py
│               ├── train.py
│               ├── score.py
│               └── README.md
│
├── machines/
│   └── inkjet_printer/              # one folder per machine
│       ├── config.yaml              # Pipeline #1 contract: input schema (numerical / categorical / targets) + output_dir. DataFrame-tied, not machine-tied.
│       ├── recovery_methods.yaml    # legacy cheat sheet (being replaced by RAG against data/RAG/)
│       └── output/
│           ├── sample_shop_floor_slice.csv  # test input — represents what the trigger will pull from the shop-floor DB
│           ├── golden_dataframe.csv         # Pipeline #1 output
│           ├── preprocessor.pkl             # fitted ColumnTransformer (saved with --save-pipeline)
│           ├── context_model.pkl
│           └── context_model_impacts.json
│
├── .gitignore
└── README.md
```

Skeletons in `pcil/trigger.py`, `pcil/rag/`, and `pcil/utils/anomaly/`
are wired end-to-end (CLIs run, package layout is final) but the core
implementation functions raise `NotImplementedError` — see each
package's `README.md` for the TODO list a teammate can pick up.

The raw machine data lives **outside** this repo (it's too big for
GitHub). See the Setup section below for where to put it.

---

## Setup

This repo holds code, configs, and small generated artefacts only. The
upstream shop-floor database isn't real yet — for now Pipeline #1 reads
a CSV-shaped slice instead. A sample slice is committed at
`machines/inkjet_printer/output/sample_shop_floor_slice.csv` (this is
the Week-1 Golden DataFrame, kept so the refactored pipeline has a
real input to run against).

When the shop-floor DB exists, this same `--input` argument will be
replaced by a Postgres pull triggered by `trigger.py`.

---

## Quickstart

From the repo root:

```bash
# 1. Run Pipeline #1: shop-floor slice -> Golden DataFrame
python pcil/preprocess.py --input machines/inkjet_printer/output/sample_shop_floor_slice.csv

# 2. Run the adapter demo (validates the Golden DataFrame, prints X/y shapes)
python pcil/adapter.py

# 3. Train the Context Model (writes context_model.pkl and impacts JSON)
python pcil/train_context_model.py
```

`preprocess.py` accepts:

```bash
python pcil/preprocess.py --input slice.csv                          # default config (inkjet_printer)
python pcil/preprocess.py --input slice.csv --config inkjet_printer  # by machine name
python pcil/preprocess.py --input slice.csv --config path/to/config.yaml
python pcil/preprocess.py --input slice.csv --save-pipeline          # also persist preprocessor.pkl
```

`train_context_model.py` accepts the same machine-name / YAML-path
arguments as before.

---

## Adding a new machine

1. Copy `machines/inkjet_printer/` to `machines/<your_machine>/`. Each
   machine still gets its own `output/` directory for trained
   artefacts.
2. Edit `config.yaml` if its column lists differ from the inkjet's. The
   YAML now describes only:
   - `input.timestamp_column`
   - `input.numerical_features` — columns scaled to [0, 1] via MinMaxScaler
   - `input.categorical_features` — columns one-hot-encoded via OneHotEncoder
   - `input.targets` — passed through unchanged
   - `pipeline.output_dir`
3. Run the same three commands as above with `--config <your_machine>`.

The code never changes. The YAML drives everything. Note: the YAML is
DataFrame-tied (it describes the shop-floor slice), not machine-tied —
multiple machines feeding into the same shop-floor schema can share
one config.yaml.

---

## Dependencies

Python 3.13+ with:

```bash
pip install pandas numpy pyyaml scikit-learn joblib tqdm matplotlib reportlab
```

---

## Status (Week 2, 8 May 2026)

| Pipeline | Status |
|---|---|
| #1 Pre-processing | refactored — `sklearn.pipeline.Pipeline` + `ColumnTransformer`, accepts shop-floor CSV slice, drops legacy `scenario` column. 625-row × 11-col Golden DataFrame on the inkjet sample. |
| Adapter | unchanged — validates schema + range, returns numpy arrays |
| #2 Context Model | v0 working — multi-target LinearRegression |
| #3 LLM | not started; RAG against `data/RAG/` (Winardi's 7 reference DOCX files) is the Week-2 prototype task |
