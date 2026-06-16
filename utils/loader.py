"""
Data loading & feature preparation for the CPS enrollment forecasting app.

The CSV `CPS_Enrollment_Forecasting.csv` is the leak-free feature table emitted
by the pipeline notebook — one row per school × grade × school-year with the
ENROLLMENT label and every model feature except SCHOOL_EFFECT, which we
reconstruct here exactly as the notebook does (train-only target encoding).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from utils.forecast import GRADE_IDX, GRADE_IDX_KEEP, FEATURES, predict_rows

CSV_PATH = "CPS_Enrollment_Forecasting.csv"

# Training window used in the notebook (Section 1 config: train 2020-2025,
# validate 2026, forecast 2027). SCHOOL_EFFECT and the median imputations must
# be computed from these years only, exactly as in the notebook.
TRAIN_START_YEAR = 2020
TRAIN_END_YEAR = 2025
SMOOTHING = 20.0

# Notebook Section 9 — NULL imputation columns (train-only medians)
NULL_FILL_COLS = [
    "SAME_GRADE_2YR_AGO",
    "FEEDER_GRADE_LAST_YEAR",
    "FEEDER_GRADE_2YR_AGO",
    "COHORT_SURVIVAL_RATE",
    "AVG_SURVIVAL_RATE_3YR",
    "DISTRICT_GRADE_ENROLLMENT_LAST_YEAR",
    "SCHOOL_TOTAL_LAST_YEAR",
]

GOV_LABELS = {
    "District": "District-run — traditional CPS neighbourhood and magnet schools.",
    "Charter":  "Independently operated, publicly funded schools with curriculum autonomy.",
    "Contract": "Privately operated under contract with CPS for specific programs.",
    "ALOP":     "Alternative Learning Opportunities — re-engagement & continuation schools.",
    "SAFE":     "Safe schools / specialised support settings.",
}


@st.cache_data(ttl=3600)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)

    numeric = [
        "GRADE_NUMERIC", "SCHOOL_YEAR", "ENROLLMENT",
        "SAME_GRADE_LAST_YEAR", "SAME_GRADE_2YR_AGO",
        "FEEDER_GRADE_LAST_YEAR", "FEEDER_GRADE_2YR_AGO", "HAS_FEEDER_GRADE",
        "SCHOOL_TOTAL_LAST_YEAR", "COHORT_SURVIVAL_RATE", "AVG_SURVIVAL_RATE_3YR",
        "DISTRICT_GRADE_ENROLLMENT_LAST_YEAR", "IS_MIGRANT_ANOMALY_YEAR",
        "GOVERNANCE_ENCODED", "IS_SELECTIVE", "IS_ATTENDANCE_AREA",
        "IS_SMALL_SCHOOL", "IS_HIGH_SCHOOL", "REGION_ENCODED", "IS_SCHOOL_OPEN",
    ]
    for c in numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # GRADE_idx — StringIndexer mapping (unseen -> keep bucket)
    df["GRADE_idx"] = df["GRADE"].astype(str).map(GRADE_IDX).fillna(GRADE_IDX_KEEP)

    # ── SCHOOL_EFFECT: train-only smoothed mean cohort-survival-rate ──────────
    train = df[df["SCHOOL_YEAR"] <= TRAIN_END_YEAR]
    global_sr = train.loc[train["COHORT_SURVIVAL_RATE"].notna(),
                          "COHORT_SURVIVAL_RATE"].mean()
    global_sr = float(global_sr) if pd.notna(global_sr) else 1.0
    stats = (train[train["COHORT_SURVIVAL_RATE"].notna()]
             .groupby("SCHOOL_KEY")["COHORT_SURVIVAL_RATE"].agg(["sum", "count"]))
    school_effect = (stats["sum"] + SMOOTHING * global_sr) / (stats["count"] + SMOOTHING)
    df["SCHOOL_EFFECT"] = df["SCHOOL_KEY"].map(school_effect).fillna(global_sr)
    df.attrs["global_sr"] = global_sr

    # ── Median imputation: train-only medians, applied everywhere ─────────────
    # (notebook Section 9 — prevents leakage from val/forecast into the fills)
    median_fills = {c: float(train[c].median()) for c in NULL_FILL_COLS}
    df[NULL_FILL_COLS] = df[NULL_FILL_COLS].fillna(median_fills)
    df.attrs["median_fills"] = median_fills

    # ── Friendly labels ──────────────────────────────────────────────────────
    df["GOVERNANCE"] = df["GOVERNANCE"].fillna("Unknown")
    # NETWORK — the real CPS multi-school grouping that schools roll up into.
    df["NETWORK"] = df["NETWORK"].fillna("Unassigned") if "NETWORK" in df.columns else "Unassigned"
    # REGION is kept internally only to back the REGION_ENCODED model feature.
    region_name = (df.dropna(subset=["ANNUAL_REGIONAL_ANALYSIS_REGION"])
                   .groupby("REGION_ENCODED")["ANNUAL_REGIONAL_ANALYSIS_REGION"]
                   .agg(lambda s: s.mode().iloc[0]))
    df["REGION"] = df["REGION_ENCODED"].map(region_name)
    df["REGION"] = df["REGION"].fillna(
        df["ANNUAL_REGIONAL_ANALYSIS_REGION"]).fillna("Unknown Region")

    df["SCHOOL_LABEL"] = "School " + df["SCHOOL_KEY"].astype(int).astype(str)
    return df


def latest_year(df: pd.DataFrame) -> int:
    return int(df["SCHOOL_YEAR"].max())


def with_network(fc: pd.DataFrame, _df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Guarantee a NETWORK column on a forecast frame.

    Self-heals against a stale `build_forecast` cache (e.g. a frame computed
    before NETWORK existed): if NETWORK is missing, it is backfilled from the
    source table by school, falling back to "Unassigned".
    """
    if "NETWORK" in fc.columns:
        return fc
    df = load_data() if _df is None else _df
    fc = fc.copy()
    if "NETWORK" in df.columns:
        net = (df.dropna(subset=["NETWORK"]).groupby("SCHOOL_KEY")["NETWORK"].first())
        fc["NETWORK"] = fc["SCHOOL_KEY"].map(net).fillna("Unassigned")
    else:
        fc["NETWORK"] = "Unassigned"
    return fc


# ════════════════════════════════════════════════════════════════════════════
# Next-year forecast (roll features forward from the latest actual year)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner="Building next-year enrollment forecast…")
def build_forecast(_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One forecast row per open school × grade for the next school year.

    Constructs each grade's next-year feature vector by rolling the same-grade
    and feeder-grade lags forward from the two most recent actual years, then
    scores it with the GBT. Returns a tidy frame with FORECAST_ENROLLMENT and
    the most-recent actual for comparison.
    """
    df = load_data() if _df is None else _df
    latest = latest_year(df)
    prev = latest - 1
    fyear = latest + 1
    global_sr = df.attrs.get("global_sr", 1.0)

    enr = {(int(r.SCHOOL_KEY), int(r.GRADE_NUMERIC), int(r.SCHOOL_YEAR)): float(r.ENROLLMENT)
           for r in df.itertuples(index=False)}
    school_total = (df[df["SCHOOL_YEAR"] == latest]
                    .groupby("SCHOOL_KEY")["ENROLLMENT"].sum().to_dict())
    dist_grade = (df[df["SCHOOL_YEAR"] == latest]
                  .groupby("GRADE_NUMERIC")["ENROLLMENT"].sum().to_dict())

    base = df[(df["SCHOOL_YEAR"] == latest) & (df["IS_SCHOOL_OPEN"] == 1)].copy()

    rows = []
    for r in base.itertuples(index=False):
        s, gn = int(r.SCHOOL_KEY), int(r.GRADE_NUMERIC)
        same_last = enr.get((s, gn, latest), np.nan)
        same_2yr  = enr.get((s, gn, prev), np.nan)
        feed_last = enr.get((s, gn - 1, latest), np.nan)
        feed_2yr  = enr.get((s, gn - 1, prev), np.nan)
        has_feeder = 1.0 if pd.notna(feed_last) else 0.0

        if pd.notna(feed_2yr) and feed_2yr > 0:
            csr = same_last / feed_2yr
        else:
            csr = r.COHORT_SURVIVAL_RATE if pd.notna(r.COHORT_SURVIVAL_RATE) else global_sr

        rows.append({
            "SCHOOL_KEY": s,
            "GRADE": r.GRADE,
            "GRADE_NUMERIC": gn,
            "NETWORK": r.NETWORK,
            "REGION": r.REGION,
            "GOVERNANCE": r.GOVERNANCE,
            "SCHOOL_LABEL": r.SCHOOL_LABEL,
            "ACTUAL_LATEST": same_last,
            # ── model features ──
            "SAME_GRADE_LAST_YEAR": same_last,
            "SAME_GRADE_2YR_AGO": same_2yr,
            "FEEDER_GRADE_LAST_YEAR": feed_last,
            "FEEDER_GRADE_2YR_AGO": feed_2yr,
            "HAS_FEEDER_GRADE": has_feeder,
            "SCHOOL_TOTAL_LAST_YEAR": school_total.get(s, np.nan),
            "COHORT_SURVIVAL_RATE": csr,
            "AVG_SURVIVAL_RATE_3YR": r.AVG_SURVIVAL_RATE_3YR,
            "DISTRICT_GRADE_ENROLLMENT_LAST_YEAR": dist_grade.get(gn, np.nan),
            "IS_MIGRANT_ANOMALY_YEAR": 0.0,
            "SCHOOL_EFFECT": r.SCHOOL_EFFECT,
            "GOVERNANCE_ENCODED": r.GOVERNANCE_ENCODED,
            "IS_SELECTIVE": r.IS_SELECTIVE,
            "IS_ATTENDANCE_AREA": r.IS_ATTENDANCE_AREA,
            "IS_SMALL_SCHOOL": r.IS_SMALL_SCHOOL,
            "IS_HIGH_SCHOOL": r.IS_HIGH_SCHOOL,
            "REGION_ENCODED": r.REGION_ENCODED,
            "GRADE_idx": r.GRADE_idx,
        })

    fc = pd.DataFrame(rows)
    # Same train-only median fills the notebook applies before scoring (Section 14)
    fc[NULL_FILL_COLS] = fc[NULL_FILL_COLS].fillna(df.attrs["median_fills"])
    fc["FORECAST_ENROLLMENT"] = predict_rows(fc).round(0)
    fc["FORECAST_YEAR"] = fyear
    return fc


@st.cache_data(ttl=3600, show_spinner=False)
def model_vs_baseline(_df: pd.DataFrame | None = None) -> dict:
    """Live model-vs-baseline accuracy on the held-out validation year.

    Baseline = the legacy Cohort Survival Rate estimate
    (FEEDER_GRADE_LAST_YEAR × AVG_SURVIVAL_RATE_3YR, the notebook's definition).
    Model    = the saved GBT scored by ``predict_rows``.

    Everything is recomputed from the live model + data, so retraining (new model
    files) or new data automatically refreshes the dashboard's improvement KPIs.
    Returns a dict keyed by metric, each with baseline / model / improvement %.
    """
    df = load_data() if _df is None else _df
    val = latest_year(df)
    v = df[(df["SCHOOL_YEAR"] == val) & (df["ENROLLMENT"] > 0)].copy()
    actual = v["ENROLLMENT"].to_numpy(float)
    model_pred = predict_rows(v)
    baseline = (v["FEEDER_GRADE_LAST_YEAR"]
                * v["AVG_SURVIVAL_RATE_3YR"].fillna(1.0)).to_numpy(float)

    mape   = lambda p: float(np.mean(np.abs((p - actual) / actual)) * 100)
    medape = lambda p: float(np.median(np.abs((p - actual) / actual)) * 100)
    mae    = lambda p: float(np.mean(np.abs(p - actual)))
    wape   = lambda p: float(np.sum(np.abs(p - actual)) / np.sum(actual) * 100)

    defs = [
        ("MAPE",   "Average % error",     "How far off the typical estimate is, on average.",       mape,   "{:.0f}%"),
        ("MedAPE", "Typical % error",     "The middle case — ignores a few extreme outliers.",       medape, "{:.0f}%"),
        ("MAE",    "Avg miss (students)", "Average students off, per school and grade.",             mae,    "{:.0f}"),
        ("WAPE",   "Budget error",        "Total over/under-count as a share of real enrollment.",   wape,   "{:.0f}%"),
    ]
    out = {"_meta": {"year": int(val), "n": int(len(v))}, "order": [d[0] for d in defs]}
    for key, label, help_, fn, fmt in defs:
        b, m = fn(baseline), fn(model_pred)
        out[key] = {"label": label, "help": help_, "fmt": fmt, "baseline": b, "model": m,
                    "improvement": ((b - m) / b * 100) if b else float("nan")}
    return out


@st.cache_data(ttl=3600)
def district_year_totals(_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """District-wide total enrollment per school-year (open schools only)."""
    df = load_data() if _df is None else _df
    out = (df[df["IS_SCHOOL_OPEN"] == 1]
           .groupby("SCHOOL_YEAR")["ENROLLMENT"].sum().reset_index())
    return out
