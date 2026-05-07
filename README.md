# PCIL — Production Context Intelligence Layer

A pipeline that watches a factory machine, figures out what is slowing it
down, and produces an operator-readable explanation. The first machine is
an inkjet printer at A*STAR SIMTech.

ITP project · Dion Ko (2401112), Zi Hin, Robin, Daniel, Jaymon · Supervisor: Winardi.

---

## What it does

```
raw machine CSVs --> Pipeline #1 --> Golden DataFrame --> Pipeline #2 --> Pipeline #3 --> dashboard
                     (preprocess)    (clean spreadsheet)  (math model)    (LLM + RAG)     (text)
```

- **Pipeline #1** turns messy CSVs (MQTT error flags, 25 kHz acoustic
  recordings, machine parameters) into a tidy 1-row-per-second
  spreadsheet where every numeric value sits between 0 and 1.
- **Pipeline #2** fits a model that says *"feature X had impact +0.4 on
  OEE"*. v0 is linear regression; SHAP comes later.
- **Pipeline #3** is the LLM stage that turns those impacts plus a
  recovery-methods cheat sheet into a sentence the operator can act on.
  Not implemented yet — the cheat sheet is.

---

## Project structure

```
PCIL/
├── pcil/                            # shared code, used by every machine
│   ├── preprocess.py                # raw CSVs -> Golden DataFrame
│   ├── adapter.py                   # Golden DataFrame -> numpy arrays for ML
│   └── train_context_model.py       # fits LinearRegression, writes impact JSON + .pkl
│
├── machines/
│   └── inkjet_printer/              # one folder per machine
│       ├── inkjet_printer.yaml      # what to read, what to extract
│       ├── recovery_methods.yaml    # troubleshooting cheat sheet
│       └── output/                  # generated artefacts
│           ├── golden_dataframe.csv
│           ├── context_model.pkl
│           └── context_model_impacts.json
│
├── .gitignore
└── README.md
```

The raw machine data lives **outside** this repo (it's too big for
GitHub). See the Setup section below for where to put it.

---

## Setup — where to put the raw data

This repo holds code, configs, and small generated artefacts only. The
raw machine recordings are not tracked here. Before running anything,
the data needs to sit **next to** your clone of PCIL, not inside it.

If you clone PCIL into `C:\projects\PCIL\`, the expected layout is:

```
C:\projects\
├── PCIL\                                  ← this repo
└── Inkjet Printer Data Collection\        ← the raw data
    ├── No Error.csv
    ├── Air Pressure Low.csv
    ├── Cycle Stop.csv
    ├── EStop Activated.csv
    ├── Printer Error.csv
    └── Acoustic Sensor Data\
        ├── machine_off_clean.csv
        ├── machine_off_anomaly.csv
        ├── machine_on_clean.csv
        └── machine_on_anomaly.csv
C:\projects\Week0\Clean_Data.csv           ← machine parameters
```

The YAML's `pipeline.base_data_dir: "../../.."` resolves to whatever
folder PCIL sits in. Adjust that line if your data lives elsewhere.

**A*STAR teammates:** the recordings are on the shared drive — ask Dion
or Winardi for the exact path.

---

## Quickstart

From the `PCIL/` root:

```bash
# 1. Generate the Golden DataFrame for the inkjet
python pcil/preprocess.py

# 2. Run the adapter demo (validates the spreadsheet, prints X/y shapes)
python pcil/adapter.py

# 3. Train the Context Model (writes context_model.pkl and impacts JSON)
python pcil/train_context_model.py
```

Each script accepts a machine name as its first argument (defaults to
`inkjet_printer`):

```bash
python pcil/preprocess.py <machine_name>
python pcil/train_context_model.py <machine_name>
```

Or an explicit YAML path:

```bash
python pcil/preprocess.py machines/inkjet_printer/inkjet_printer.yaml
```

---

## Adding a new machine

1. Copy `machines/inkjet_printer/` to `machines/<your_machine>/` and
   rename the YAML file to match the folder.
2. Edit the YAML:
   - `machine.id` and `machine.name`
   - `mqtt.data_dir`, `mqtt.scenarios`, `mqtt.flag_columns`
   - `acoustic.data_dir`, `acoustic.conditions` (or empty if no
     vibration sensor)
   - `machine_params.file`, `machine_params.columns`
   - `schema.factors` — pick the ≤6 features you want in the Golden
     DataFrame
   - `oee.scenarios` — fill in real OEE inputs
3. Replace the entries in `recovery_methods.yaml` with your machine's
   error codes.
4. Run the same three commands as above with your new machine name.

The code never changes. The YAML drives everything.

---

## Dependencies

Python 3.13+ with:

```bash
pip install pandas numpy pyyaml scikit-learn joblib tqdm matplotlib reportlab
```

---

## Status (Week 1, 7 May 2026)

| Pipeline | Status |
|---|---|
| #1 Pre-processing | working — produces 625-row × 12-col Golden DataFrame for the inkjet |
| Adapter | working — validates schema + range, returns numpy arrays |
| #2 Context Model | v0 working — multi-target LinearRegression |
| #3 LLM | not started; `recovery_methods.yaml` ready to feed RAG |


