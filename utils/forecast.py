"""
Pure-Python GBT scorer for the CPS enrollment model.

Reads the Spark MLlib GBTRegressionModel Parquet files directly — no Java /
PySpark runtime is required. The saved PipelineModel is:

    StringIndexer(GRADE -> GRADE_idx)  ->  VectorAssembler  ->  GBTRegressor

The GBT has 100 trees. Tree 0 is the initial-prediction tree (weight 1.0); the
remaining 99 trees carry the stepSize weight (0.1). Weights are read from
treesMetadata/part-*.parquet. Feature 18 (GRADE_idx) uses categorical splits;
every other split is a continuous threshold.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = "gbt_enrollment_model"

# ── Feature order — must match the VectorAssembler inputCols exactly ──────────
FEATURES = [
    "SAME_GRADE_LAST_YEAR",                  # 0
    "SAME_GRADE_2YR_AGO",                    # 1
    "FEEDER_GRADE_LAST_YEAR",                # 2
    "FEEDER_GRADE_2YR_AGO",                  # 3
    "SCHOOL_TOTAL_LAST_YEAR",                # 4
    "COHORT_SURVIVAL_RATE",                  # 5
    "AVG_SURVIVAL_RATE_3YR",                 # 6
    "DISTRICT_GRADE_ENROLLMENT_LAST_YEAR",   # 7
    "EFFECTIVE_LEADERS_LAST_YEAR",           # 8   5Essentials pillar scores (prior year)
    "COLLABORATIVE_TEACHERS_LAST_YEAR",      # 9
    "INVOLVED_FAMILIES_LAST_YEAR",           # 10
    "SUPPORTIVE_ENVIRONMENT_LAST_YEAR",      # 11
    "AMBITIOUS_INSTRUCTION_LAST_YEAR",       # 12
    "SAME_GRADE_AVG_3YR",                    # 13  multi-year averages & trends
    "SAME_GRADE_TREND",                      # 14
    "FEEDER_GRADE_TREND",                    # 15
    "SCHOOL_TOTAL_2YR_AGO",                  # 16
    "SCHOOL_TOTAL_AVG_3YR",                  # 17
    "SCHOOL_TOTAL_TREND",                    # 18
    "DISTRICT_GRADE_ENROLLMENT_2YR_AGO",     # 19
    "DISTRICT_GRADE_ENROLLMENT_AVG_3YR",     # 20
    "DISTRICT_GRADE_ENROLLMENT_TREND",       # 21
    "GRADE_idx",                             # 22  (from StringIndexer)
]

# StringIndexer labelsArray (frequencyDesc order). Index = GRADE_idx.
# Unseen grades map to len(GRADE_LABELS) because handleInvalid="keep".
GRADE_LABELS = ["8", "7", "6", "5", "4", "3", "K", "2", "1", "9", "11", "12", "10"]
GRADE_IDX = {g: i for i, g in enumerate(GRADE_LABELS)}
GRADE_IDX_KEEP = float(len(GRADE_LABELS))

# ── Model accuracy — held-out validation (train 2020-2025, validate SY2026),
# all grades, reproduced by this app's GBT scorer on the current CSV.
MODEL_MAE  = 10.9     # students, per school × grade
MODEL_RMSE = 43.8     # students, per school × grade
MODEL_R2   = 0.85

# ── Official results from the model review document (4-year walk-forward
#    backtest SY2023–2026, 18,899 school-grade predictions, 842 open schools) ─
BACKTEST_HEADLINE = {
    # metric: (CSR current SAS, ML proposed GBT)
    "Total over/under-prediction (students)": ("+193,048", "−51,357"),
    "as % of actual enrollment":              ("+27.3%",   "−7.3%"),
    "Budget error rate (WAPE)":               ("71.4%",    "24.7%"),
    "Average error per school-grade (MAE)":   ("26.8",     "9.3"),
    "Typical % error per cell (MedAPE)":      ("25.0%",    "15.6%"),
    "Variance explained (R²)":                ("−0.45",    "+0.78"),
}

BACKTEST_BY_YEAR = [
    # year, n, MAE_CSR, MAE_ML, RMSE_CSR, RMSE_ML, WAPE_CSR, WAPE_ML, R2_CSR, R2_ML
    (2023, 4696, 26.2, 7.7,  89.8, 17.5, "79.7%", "23.3%", "−2.36", "0.87"),
    (2024, 4697, 29.4, 10.7, 97.5, 38.6, "75.5%", "27.5%", "−0.87", "0.71"),
    (2025, 4757, 25.7, 10.6, 83.1, 48.2, "67.1%", "27.6%", "−0.20", "0.60"),
    (2026, 4749, 25.8, 8.1,  98.1, 33.0, "64.8%", "20.3%", "+0.06", "0.89"),
]

# Official Spark featureImportances of the saved model (review PDF, Section 4).
OFFICIAL_FI = {
    "SAME_GRADE_LAST_YEAR":                0.3203,
    "DISTRICT_GRADE_ENROLLMENT_LAST_YEAR": 0.1405,
    "SCHOOL_TOTAL_LAST_YEAR":              0.1185,
    "GRADE_idx":                           0.1107,
    "FEEDER_GRADE_LAST_YEAR":              0.0592,
    "COHORT_SURVIVAL_RATE":                0.0510,
    "FEEDER_GRADE_2YR_AGO":                0.0480,
    "SCHOOL_EFFECT":                       0.0400,
    "AVG_SURVIVAL_RATE_3YR":               0.0371,
    "SAME_GRADE_2YR_AGO":                  0.0245,
    "REGION_ENCODED":                      0.0129,
    "HAS_FEEDER_GRADE":                    0.0105,
    "GRADE_NUMERIC":                       0.0087,
    "IS_MIGRANT_ANOMALY_YEAR":             0.0058,
    "IS_SELECTIVE":                        0.0056,
    "IS_HIGH_SCHOOL":                      0.0035,
    "IS_ATTENDANCE_AREA":                  0.0014,
    "GOVERNANCE_ENCODED":                  0.0013,
    "IS_SMALL_SCHOOL":                     0.0005,
}


# ════════════════════════════════════════════════════════════════════════════
# Model I/O
# ════════════════════════════════════════════════════════════════════════════
def _read_trees_and_weights(model_dir: str) -> tuple[dict, dict]:
    """Load every tree node and its per-tree weight from the GBT stage."""
    data_files = sorted(glob.glob(
        f"{model_dir}/**/stages/*GBTRegressor*/data/part-*.parquet", recursive=True))
    if not data_files:
        raise FileNotFoundError(
            f"No GBT Parquet data found under {model_dir} (searched recursively for stages/*GBTRegressor*/data)")

    node_df = pd.concat([pd.read_parquet(f) for f in data_files], ignore_index=True)
    trees: dict[int, dict[int, dict]] = {}
    for _, row in node_df.iterrows():
        tid = int(row["treeID"])
        nd = dict(row["nodeData"])
        trees.setdefault(tid, {})[int(nd["id"])] = nd

    wfiles = sorted(glob.glob(
        f"{model_dir}/**/stages/*GBTRegressor*/treesMetadata/part-*.parquet", recursive=True))
    wmeta = pd.concat([pd.read_parquet(f) for f in wfiles], ignore_index=True)
    weights = {int(r["treeID"]): float(r["weights"]) for _, r in wmeta.iterrows()}
    return trees, weights


@st.cache_resource(show_spinner="Loading enrollment model…")
def load_gbt_model() -> tuple[dict, dict]:
    """Load the enrollment GBT once per server start. Returns (trees, weights)."""
    return _read_trees_and_weights(MODEL_DIR)


# ════════════════════════════════════════════════════════════════════════════
# Tree traversal & scoring
# ════════════════════════════════════════════════════════════════════════════
def _traverse(nodes: dict[int, dict], fvec: np.ndarray) -> float:
    node = nodes[0]
    while True:
        lc = int(node["leftChild"])
        if lc == -1:
            return float(node["prediction"])
        sp = node["split"]
        fi = int(sp["featureIndex"])
        cats = sp["leftCategoriesOrThreshold"]
        ncat = int(sp["numCategories"])
        v = fvec[fi]
        if ncat == -1:                       # continuous split
            go_left = v <= float(cats[0])
        else:                                # categorical split (GRADE_idx)
            go_left = v in {float(c) for c in cats}
        node = nodes[lc] if go_left else nodes[int(node["rightChild"])]


def score_gbt(trees: dict, weights: dict, fvec: np.ndarray) -> float:
    """Σ  tree_i(x) × weight_i   for all trees."""
    return sum(_traverse(trees[t], fvec) * weights[t] for t in trees)


def predict_rows(feature_df: pd.DataFrame) -> np.ndarray:
    """Score a DataFrame that already contains all FEATURES columns."""
    trees, weights = load_gbt_model()
    X = feature_df[FEATURES].astype(float).fillna(0.0).values
    return np.array([max(0.0, score_gbt(trees, weights, row)) for row in X])


# ════════════════════════════════════════════════════════════════════════════
# Feature importance (gain-weighted, read from the trees)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def feature_importances() -> pd.DataFrame:
    """Gini/gain feature importance, normalised to sum to 1."""
    trees, _ = load_gbt_model()
    imp = np.zeros(len(FEATURES))
    for nodes in trees.values():
        for nd in nodes.values():
            if int(nd["leftChild"]) == -1:
                continue
            fi = int(nd["split"]["featureIndex"])
            imp[fi] += float(nd["gain"]) * float(nd["rawCount"])
    if imp.sum() > 0:
        imp = imp / imp.sum()
    return (pd.DataFrame({"feature": FEATURES, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))
