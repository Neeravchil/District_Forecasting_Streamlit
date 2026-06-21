"""
Data loading & feature preparation for the CPS enrollment forecasting app.

The CSV `CPS_Enrollment_Forecasting.csv` is the leak-free feature table emitted
by the pipeline notebook — one row per school × grade × school-year with the
ENROLLMENT label and every model feature except SCHOOL_EFFECT, which we
reconstruct here exactly as the notebook does (train-only target encoding).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st

from utils.forecast import GRADE_IDX, GRADE_IDX_KEEP, FEATURES, predict_rows

# ── Business-impact assumptions (Student-Based Budgeting) ─────────────────────
PER_PUPIL_SBB = 6200    # FY26 SBB base allocation per projected pupil ($)
# The current method = cohort-survival ratio. Raw 3-yr survival ratios spike to
# 100x+ when a feeder grade is near zero — projections no real method would make —
# so we hold them to a plausible ceiling for a fair, credible baseline.
SURVIVAL_CAP = 1.5

CSV_PATH = "CPS_Enrollment_Forecasting.csv"
# Per-school display metadata (NETWORK, governance, region, community), merged in
# when the model CSV omits it — so network-level views work after a re-export.
SCHOOL_META_PATH = "school_metadata.csv"

# Training window used in the notebook (Section 1 config: train 2020-2025,
# validate 2026, forecast 2027). SCHOOL_EFFECT and the median imputations must
# be computed from these years only, exactly as in the notebook.
TRAIN_START_YEAR = 2020
TRAIN_END_YEAR = 2025
SMOOTHING = 20.0

# Notebook Section 9 — NULL imputation columns (train-only medians)
NULL_FILL_COLS = [
    "SAME_GRADE_2YR_AGO", "FEEDER_GRADE_LAST_YEAR", "FEEDER_GRADE_2YR_AGO",
    "COHORT_SURVIVAL_RATE", "AVG_SURVIVAL_RATE_3YR",
    "DISTRICT_GRADE_ENROLLMENT_LAST_YEAR", "SCHOOL_TOTAL_LAST_YEAR",
    "EFFECTIVE_LEADERS_LAST_YEAR", "COLLABORATIVE_TEACHERS_LAST_YEAR",
    "INVOLVED_FAMILIES_LAST_YEAR", "SUPPORTIVE_ENVIRONMENT_LAST_YEAR",
    "AMBITIOUS_INSTRUCTION_LAST_YEAR",
    "SAME_GRADE_AVG_3YR", "SAME_GRADE_TREND", "FEEDER_GRADE_TREND",
    "SCHOOL_TOTAL_2YR_AGO", "SCHOOL_TOTAL_AVG_3YR", "SCHOOL_TOTAL_TREND",
    "DISTRICT_GRADE_ENROLLMENT_2YR_AGO", "DISTRICT_GRADE_ENROLLMENT_AVG_3YR",
    "DISTRICT_GRADE_ENROLLMENT_TREND",
]

# ── Entry-grade (K & 9) XGBoost model ─────────────────────────────────────────
# The saved Spark GBT was never trained on K/9, so we score those two grades with
# the dedicated XGBoost model from XGBoost_KG_Grade9_Training.ipynb. Feature list
# and order must match that notebook exactly; NaNs are preserved (XGBoost handles
# missing values natively, unlike the median-filled Spark path).
XGB_MODEL_PATH = "xgb_model_k9.pkl"
ENTRY_GRADES = ("K", "9")
XGB_FEATURES = [
    "SAME_GRADE_LAST_YEAR", "SAME_GRADE_2YR_AGO", "FEEDER_GRADE_LAST_YEAR",
    "FEEDER_GRADE_2YR_AGO", "SCHOOL_TOTAL_LAST_YEAR", "COHORT_SURVIVAL_RATE",
    "AVG_SURVIVAL_RATE_3YR", "DISTRICT_GRADE_ENROLLMENT_LAST_YEAR",
    "IS_MIGRANT_ANOMALY_YEAR",
    "EFFECTIVE_LEADERS_LAST_YEAR", "COLLABORATIVE_TEACHERS_LAST_YEAR",
    "INVOLVED_FAMILIES_LAST_YEAR", "SUPPORTIVE_ENVIRONMENT_LAST_YEAR",
    "AMBITIOUS_INSTRUCTION_LAST_YEAR",
    "SAME_GRADE_AVG_3YR", "SAME_GRADE_TREND", "FEEDER_GRADE_TREND",
    "SCHOOL_TOTAL_2YR_AGO", "SCHOOL_TOTAL_AVG_3YR", "SCHOOL_TOTAL_TREND",
    "DISTRICT_GRADE_ENROLLMENT_2YR_AGO", "DISTRICT_GRADE_ENROLLMENT_AVG_3YR",
    "DISTRICT_GRADE_ENROLLMENT_TREND",
]


@st.cache_resource(show_spinner=False)
def load_xgb_model():
    """Load the saved XGBoost entry-grade model.

    No fallback: if the model cannot be loaded (missing file, xgboost not
    installed, etc.) the error propagates — K & 9 must be scored by this model.
    """
    import pickle
    with open(XGB_MODEL_PATH, "rb") as fh:
        return pickle.load(fh)


@st.cache_data(ttl=3600, show_spinner=False)
def xgb_k9_forecast() -> dict:
    """Next-year K & 9 predictions keyed by (SCHOOL_KEY, GRADE).

    Built from the RAW CSV with NaNs preserved — exactly as the training notebook
    feeds the model — never the median-filled forecast frame. No fallback: any
    load/predict failure propagates as an error.
    """
    model = load_xgb_model()
    raw = pd.read_csv(CSV_PATH)
    raw["GRADE"] = raw["GRADE"].astype(str)
    if "IS_SCHOOL_OPEN" not in raw.columns:
        raw["IS_SCHOOL_OPEN"] = 1
    _y = pd.to_numeric(raw["SCHOOL_YEAR"], errors="coerce")
    raw["IS_MIGRANT_ANOMALY_YEAR"] = ((_y >= 2022) & (_y <= 2024)).astype(int)
    for c in XGB_FEATURES + ["SCHOOL_YEAR", "IS_SCHOOL_OPEN"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    latest = int(raw["SCHOOL_YEAR"].max())
    k9 = raw[(raw["SCHOOL_YEAR"] == latest) & (raw["IS_SCHOOL_OPEN"] == 1)
             & (raw["GRADE"].isin(ENTRY_GRADES))].copy()
    k9["IS_MIGRANT_ANOMALY_YEAR"] = 0
    preds = np.clip(np.round(model.predict(k9[XGB_FEATURES]), 0), 0, None)
    return {(int(s), g): float(p)
            for s, g, p in zip(k9["SCHOOL_KEY"], k9["GRADE"], preds)}


def xgb_predict_row(features: dict) -> float:
    """Single K/9 prediction from the XGBoost model — used by the Custom School
    Simulator. Any feature absent from ``features`` (e.g. SCHOOL_KEY for a
    hypothetical school) is passed as NaN, which the model handles natively.
    No fallback — load/predict errors propagate.
    """
    model = load_xgb_model()
    X = pd.DataFrame([{c: features.get(c, np.nan) for c in XGB_FEATURES}])
    return float(np.clip(np.round(model.predict(X)[0], 0), 0, None))

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

    # ── Display-only metadata (NETWORK, governance, region, community) ────────
    # The model never used these. If the model CSV omits them, merge them from the
    # committed per-school lookup so the network / governance / region breakdowns
    # still work; the migrant flag and open status are derived. Anything still
    # missing falls back to a safe default below.
    _meta = ["NETWORK", "GOVERNANCE", "ANNUAL_REGIONAL_ANALYSIS_REGION",
             "COMMUNITY", "REGION_ENCODED"]
    _need = [c for c in _meta if c not in df.columns]
    if _need and os.path.exists(SCHOOL_META_PATH):
        _lut = pd.read_csv(SCHOOL_META_PATH)
        df = df.merge(_lut[["SCHOOL_KEY"] + [c for c in _need if c in _lut.columns]],
                      on="SCHOOL_KEY", how="left")
    if "REGION_ENCODED" in df.columns:
        df["REGION_ENCODED"] = pd.to_numeric(df["REGION_ENCODED"], errors="coerce").fillna(0)

    if "IS_SCHOOL_OPEN" not in df.columns:
        df["IS_SCHOOL_OPEN"] = 1
    if "IS_MIGRANT_ANOMALY_YEAR" not in df.columns:
        _y = pd.to_numeric(df["SCHOOL_YEAR"], errors="coerce")
        df["IS_MIGRANT_ANOMALY_YEAR"] = ((_y >= 2022) & (_y <= 2024)).astype(int)
    if "HAS_FEEDER_GRADE" not in df.columns:
        df["HAS_FEEDER_GRADE"] = (df["FEEDER_GRADE_LAST_YEAR"].notna().astype(int)
                                  if "FEEDER_GRADE_LAST_YEAR" in df.columns else 0)
    if "GOVERNANCE" not in df.columns:
        df["GOVERNANCE"] = "Unknown"
    if "NETWORK" not in df.columns:
        df["NETWORK"] = "Unassigned"
    if "ANNUAL_REGIONAL_ANALYSIS_REGION" not in df.columns:
        df["ANNUAL_REGIONAL_ANALYSIS_REGION"] = np.nan
    if "COMMUNITY" not in df.columns:
        df["COMMUNITY"] = "—"
    for _c in ["GOVERNANCE_ENCODED", "IS_SELECTIVE", "IS_ATTENDANCE_AREA",
               "IS_SMALL_SCHOOL", "IS_HIGH_SCHOOL", "REGION_ENCODED"]:
        if _c not in df.columns:
            df[_c] = 0

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

    # SCHOOL_LABEL — must be a real school name. NO school-key fallback: if the
    # export has no name column, or any name is blank, raise so the gap is fixed
    # at the source (re-run the notebook's Stage 1, which writes SCHOOL_NAME).
    name_col = next((c for c in ["SCHOOL_NAME", "SCHOOL_LONG_NAME", "LONG_NAME",
                                 "SCHOOL_NM", "NAME"] if c in df.columns), None)
    if name_col is None:
        raise KeyError(
            "No school-name column in CPS_Enrollment_Forecasting.csv "
            "(looked for SCHOOL_NAME / SCHOOL_LONG_NAME / LONG_NAME / SCHOOL_NM / NAME). "
            "Re-run the notebook's Stage 1 — Cell 17 now rebuilds the feature table AND "
            "re-exports the CSV with SCHOOL_NAME — then commit the new CSV. "
            "No school-key fallback is used.")
    nm = df[name_col].astype("string").str.strip()
    blank = nm.isna() | (nm == "") | (nm.str.lower() == "nan")
    if blank.any():
        missing_keys = sorted(df.loc[blank, "SCHOOL_KEY"].astype(int).unique())[:10]
        raise ValueError(
            f"{int(blank.sum())} rows have a blank '{name_col}' "
            f"(e.g. SCHOOL_KEY {missing_keys}). Fix the school name at the source — "
            "no school-key fallback is used.")
    df["SCHOOL_LABEL"] = nm
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

    Rolls every lag / 3-year-average / year-over-year-trend feature forward from
    the most recent actual years, carries the latest 5Essentials pillar scores, then
    scores with the GBT (and the XGBoost entry-grade model for K & 9). Returns
    FORECAST_ENROLLMENT plus the most recent actual for comparison.
    """
    df = load_data() if _df is None else _df
    latest = latest_year(df)
    prev = latest - 1
    prev2 = latest - 2
    fyear = latest + 1
    global_sr = df.attrs.get("global_sr", 1.0)

    enr = {(int(r.SCHOOL_KEY), int(r.GRADE_NUMERIC), int(r.SCHOOL_YEAR)): float(r.ENROLLMENT)
           for r in df.itertuples(index=False)}

    def totals(col, year):
        return df[df["SCHOOL_YEAR"] == year].groupby(col)["ENROLLMENT"].sum().to_dict()
    school_total = {y: totals("SCHOOL_KEY", y) for y in (latest, prev, prev2)}
    dist_grade = {y: totals("GRADE_NUMERIC", y) for y in (latest, prev, prev2)}

    def mean3(vals):
        v = [x for x in vals if pd.notna(x)]
        return float(np.mean(v)) if v else np.nan

    def diff(a, b):
        return float(a - b) if pd.notna(a) and pd.notna(b) else np.nan

    base = df[(df["SCHOOL_YEAR"] == latest) & (df["IS_SCHOOL_OPEN"] == 1)].copy()
    rows = []
    for r in base.itertuples(index=False):
        s, gn = int(r.SCHOOL_KEY), int(r.GRADE_NUMERIC)
        same_last = enr.get((s, gn, latest), np.nan)
        same_2yr = enr.get((s, gn, prev), np.nan)
        same_3yr = enr.get((s, gn, prev2), np.nan)
        feed_last_raw = enr.get((s, gn - 1, latest), np.nan)
        feed_2yr_raw = enr.get((s, gn - 1, prev), np.nan)
        has_feeder = 1.0 if pd.notna(feed_last_raw) else 0.0
        feed_last = feed_last_raw if pd.notna(feed_last_raw) else same_last
        feed_2yr = feed_2yr_raw if pd.notna(feed_2yr_raw) else same_2yr
        st_last, st_2yr, st_3yr = (school_total[latest].get(s, np.nan),
                                   school_total[prev].get(s, np.nan),
                                   school_total[prev2].get(s, np.nan))
        dg_last, dg_2yr, dg_3yr = (dist_grade[latest].get(gn, np.nan),
                                   dist_grade[prev].get(gn, np.nan),
                                   dist_grade[prev2].get(gn, np.nan))
        if pd.notna(feed_2yr) and feed_2yr > 0:
            csr = same_last / feed_2yr
        else:
            csr = r.COHORT_SURVIVAL_RATE if pd.notna(r.COHORT_SURVIVAL_RATE) else global_sr

        rows.append({
            "SCHOOL_KEY": s, "GRADE": r.GRADE, "GRADE_NUMERIC": gn,
            "NETWORK": r.NETWORK, "REGION": r.REGION, "GOVERNANCE": r.GOVERNANCE,
            "SCHOOL_LABEL": r.SCHOOL_LABEL, "ACTUAL_LATEST": same_last,
            # ── model features (rolled forward to the forecast year) ──
            "SAME_GRADE_LAST_YEAR": same_last,
            "SAME_GRADE_2YR_AGO": same_2yr,
            "FEEDER_GRADE_LAST_YEAR": feed_last,
            "FEEDER_GRADE_2YR_AGO": feed_2yr,
            "SCHOOL_TOTAL_LAST_YEAR": st_last,
            "COHORT_SURVIVAL_RATE": csr,
            "AVG_SURVIVAL_RATE_3YR": r.AVG_SURVIVAL_RATE_3YR,
            "DISTRICT_GRADE_ENROLLMENT_LAST_YEAR": dg_last,
            "EFFECTIVE_LEADERS_LAST_YEAR": r.EFFECTIVE_LEADERS_LAST_YEAR,
            "COLLABORATIVE_TEACHERS_LAST_YEAR": r.COLLABORATIVE_TEACHERS_LAST_YEAR,
            "INVOLVED_FAMILIES_LAST_YEAR": r.INVOLVED_FAMILIES_LAST_YEAR,
            "SUPPORTIVE_ENVIRONMENT_LAST_YEAR": r.SUPPORTIVE_ENVIRONMENT_LAST_YEAR,
            "AMBITIOUS_INSTRUCTION_LAST_YEAR": r.AMBITIOUS_INSTRUCTION_LAST_YEAR,
            "SAME_GRADE_AVG_3YR": mean3([same_last, same_2yr, same_3yr]),
            "SAME_GRADE_TREND": diff(same_last, same_2yr),
            "FEEDER_GRADE_TREND": diff(feed_last, feed_2yr),
            "SCHOOL_TOTAL_2YR_AGO": st_2yr,
            "SCHOOL_TOTAL_AVG_3YR": mean3([st_last, st_2yr, st_3yr]),
            "SCHOOL_TOTAL_TREND": diff(st_last, st_2yr),
            "DISTRICT_GRADE_ENROLLMENT_2YR_AGO": dg_2yr,
            "DISTRICT_GRADE_ENROLLMENT_AVG_3YR": mean3([dg_last, dg_2yr, dg_3yr]),
            "DISTRICT_GRADE_ENROLLMENT_TREND": diff(dg_last, dg_2yr),
            "IS_MIGRANT_ANOMALY_YEAR": 0.0,
            "HAS_FEEDER_GRADE": has_feeder,
            "GRADE_idx": r.GRADE_idx,
        })

    fc = pd.DataFrame(rows)
    # Same train-only median fills the notebook applies before scoring
    fc[NULL_FILL_COLS] = fc[NULL_FILL_COLS].fillna(df.attrs["median_fills"])
    fc["FORECAST_ENROLLMENT"] = predict_rows(fc).round(0)

    # K & 9 only: replace the GBT value with the dedicated XGBoost model.
    is_k9 = fc["GRADE"].astype(str).isin(ENTRY_GRADES)
    if is_k9.any():
        k9 = xgb_k9_forecast()
        fc.loc[is_k9, "FORECAST_ENROLLMENT"] = [
            k9[(int(s), str(g))] for s, g in zip(
                fc.loc[is_k9, "SCHOOL_KEY"], fc.loc[is_k9, "GRADE"])]

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


@st.cache_data(ttl=3600, show_spinner=False)
def business_impact(_df: pd.DataFrame | None = None) -> dict:
    """Dollar value of the model on the held-out year: how much enrollment-driven
    budget it re-targets vs the current cohort-survival method. This is allocation
    accuracy — funding reaching the right schools — not new revenue. Recomputed
    live, so it stays correct when the model or data changes.
    """
    df = load_data() if _df is None else _df
    val = latest_year(df)
    v = df[(df["SCHOOL_YEAR"] == val) & (df["ENROLLMENT"] > 0)].copy()
    actual = v["ENROLLMENT"].to_numpy(float)
    model = np.asarray(predict_rows(v), float)
    current = (v["FEEDER_GRADE_LAST_YEAR"]
               * v["AVG_SURVIVAL_RATE_3YR"].fillna(1.0).clip(upper=SURVIVAL_CAP)).to_numpy(float)

    cur_students = float(np.abs(current - actual).sum())   # total mis-forecast (over+under)
    mdl_students = float(np.abs(model - actual).sum())
    n = len(v)
    return {
        "year": int(val), "n": int(n), "per_pupil": PER_PUPIL_SBB,
        "actual_total": float(actual.sum()),
        "cur_students": cur_students, "mdl_students": mdl_students,
        "saved_students": cur_students - mdl_students,
        "cur_dollars": cur_students * PER_PUPIL_SBB,
        "mdl_dollars": mdl_students * PER_PUPIL_SBB,
        "saved_dollars": (cur_students - mdl_students) * PER_PUPIL_SBB,
        "cur_mae": cur_students / n, "mdl_mae": mdl_students / n,
        "accuracy_x": (cur_students / mdl_students) if mdl_students else float("nan"),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def _xgb_entry_valyear() -> dict:
    """XGBoost K & 9 predictions for the held-out validation (= latest) year,
    keyed by (SCHOOL_KEY, GRADE).

    Like :func:`xgb_k9_forecast` but WITHOUT the ``IS_SCHOOL_OPEN == 1`` filter,
    so it covers every entry-grade row that the accuracy backtest evaluates
    (which includes closed schools that still reported enrollment). Built from the
    raw CSV with NaNs preserved. No fallback — load/predict failures propagate.
    """
    model = load_xgb_model()
    raw = pd.read_csv(CSV_PATH)
    raw["GRADE"] = raw["GRADE"].astype(str)
    _y = pd.to_numeric(raw["SCHOOL_YEAR"], errors="coerce")
    raw["IS_MIGRANT_ANOMALY_YEAR"] = ((_y >= 2022) & (_y <= 2024)).astype(int)
    for c in XGB_FEATURES + ["SCHOOL_YEAR"]:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    val = int(raw["SCHOOL_YEAR"].max())
    rk = raw[(raw["SCHOOL_YEAR"] == val) & raw["GRADE"].isin(ENTRY_GRADES)].copy()
    rk["IS_MIGRANT_ANOMALY_YEAR"] = 0
    rk = rk.drop_duplicates(["SCHOOL_KEY", "GRADE"], keep="last")
    preds = np.clip(np.round(model.predict(rk[XGB_FEATURES]), 0), 0, None)
    return {(int(s), str(g)): float(p)
            for s, g, p in zip(rk["SCHOOL_KEY"], rk["GRADE"], preds)}


@st.cache_data(ttl=3600)
def accuracy_by_group(group_col: str, _df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-group CSR vs ML error on the held-out validation year.

    Same definitions as :func:`model_vs_baseline` (CSR = FEEDER × survival;
    ML = the served forecast), but broken out by ``group_col`` — e.g. ``"GRADE"``
    or ``"NETWORK"``. ML uses the exact hybrid the app serves: the XGBoost
    entry-grade model for K & 9, the Spark GBT for every other grade — so the
    entry grades are scored the way users actually see them. One row per group
    with CSR/ML MAE and MedAPE plus the sample count ``n``.
    """
    df = load_data() if _df is None else _df
    val = latest_year(df)
    v = df[(df["SCHOOL_YEAR"] == val) & (df["ENROLLMENT"] > 0)].copy()

    ml = np.asarray(predict_rows(v), dtype=float).copy()
    is_k9 = v["GRADE"].astype(str).isin(ENTRY_GRADES).to_numpy()
    if is_k9.any():
        # Held-out K/9 scored by the XGBoost entry-grade model over ALL val-year
        # entry rows (incl. closed-but-enrolled schools the backtest evaluates),
        # so every row is covered. No fallback — load/predict failures propagate.
        k9 = _xgb_entry_valyear()
        ml[is_k9] = [k9[(int(s), str(g))]
                     for s, g in zip(v.loc[is_k9, "SCHOOL_KEY"], v.loc[is_k9, "GRADE"])]

    csr = (v["FEEDER_GRADE_LAST_YEAR"]
           * v["AVG_SURVIVAL_RATE_3YR"].fillna(1.0)).to_numpy(float)
    v = v.assign(_actual=v["ENROLLMENT"].to_numpy(float), _ml=ml, _csr=csr)

    recs = []
    for key, g in v.groupby(group_col, observed=True):
        a = g["_actual"].to_numpy()
        rec = {group_col: key, "n": int(len(g))}
        for nm, p in (("csr", g["_csr"].to_numpy()), ("ml", g["_ml"].to_numpy())):
            rec[f"{nm}_mae"] = float(np.mean(np.abs(p - a)))
            rec[f"{nm}_medape"] = float(np.median(np.abs((p - a) / a)) * 100)
        recs.append(rec)
    return pd.DataFrame(recs)


@st.cache_data(ttl=3600)
def district_year_totals(_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """District-wide total enrollment per school-year (open schools only)."""
    df = load_data() if _df is None else _df
    out = (df[df["IS_SCHOOL_OPEN"] == 1]
           .groupby("SCHOOL_YEAR")["ENROLLMENT"].sum().reset_index())
    return out
