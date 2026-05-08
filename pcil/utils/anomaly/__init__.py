"""
Anomaly detection utilities.

Each data type lives in its own subpackage with the same four-step structure:
    slice.py     — turn a raw stream into discrete units (cycles or windows)
    features.py  — turn each unit into a fixed-size feature vector
    model.py     — candidate anomaly models (z-score, IF, OCSVM, autoencoder)
    train.py     — CLI: dataset -> trained .pkl
    score.py     — CLI: data + .pkl -> anomaly_score column

Shared:
    normalise.py — per-machine z-score helper fitted inside each trained bundle
"""
