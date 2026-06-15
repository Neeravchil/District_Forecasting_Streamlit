"""
Page 1 — Overview.

High-level summary of the enrollment-forecasting project for the enrollment
meeting: what the model does, the headline next-year numbers at the network and
school level, and the coverage / data notes that answer the K1 & K9 and
missing-lag-feature questions.
"""

import pandas as pd
import streamlit as st

from utils.loader import load_data, build_forecast, district_year_totals, latest_year
from utils.forecast import MODEL_MAE, MODEL_R2, BACKTEST_HEADLINE

# ── Data & forecast ───────────────────────────────────────────────────────────
df = load_data()
fc = build_forecast(df)
yr_totals = district_year_totals(df)
LATEST = latest_year(df)
FYEAR = LATEST + 1

network_latest   = int(df[df["SCHOOL_YEAR"] == LATEST]["ENROLLMENT"].sum())
network_forecast = int(fc["FORECAST_ENROLLMENT"].sum())
change_pct       = (network_forecast - network_latest) / network_latest * 100
n_schools        = int(fc["SCHOOL_KEY"].nunique())
n_networks       = int(fc["REGION"].nunique())

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='background:linear-gradient(135deg,#003057 0%,#00497a 100%);
            border-radius:12px; padding:40px 48px; margin-bottom:28px;
            border-left:6px solid #C8973A; width:100%;'>
    <div style='font-size:0.8rem; font-weight:700; color:#C8973A; letter-spacing:0.12em;
                text-transform:uppercase; margin-bottom:10px;'>Project Overview</div>
    <div style='font-size:2.2rem; font-weight:800; color:#FFFFFF; line-height:1.2; margin-bottom:16px;'>
        A Machine-Learning Enrollment Forecast for {FYEAR}
    </div>
    <div style='font-size:0.97rem; color:#B8CFDF; line-height:1.7;'>
        This dashboard shows where the enrollment model stands for the {FYEAR} planning cycle.
        It replaces the legacy Cohort-Survival-Rate (CSR) spreadsheet with a Gradient-Boosted-Tree
        model trained on <b style='color:#FFFFFF;'>seven years of school&nbsp;×&nbsp;grade records</b>,
        and produces projections at both the <b style='color:#FFFFFF;'>network</b> and
        <b style='color:#FFFFFF;'>individual-school</b> level.
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI strip ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
kpis = [
    (f"{network_forecast:,}", f"{FYEAR} Network Forecast", "Total students across the whole network"),
    (f"{change_pct:+.1f}%",   "Projected Change",          f"{FYEAR} vs {LATEST} actual"),
    (f"{n_schools:,}",        "Schools Projected",         f"Across {n_networks} networks, by grade"),
    (f"~{MODEL_MAE:.0f}",     "Typical Error (MAE)",       f"Students per school-grade · R² {MODEL_R2:.2f}"),
]
for col, (val, lbl, sub) in zip([c1, c2, c3, c4], kpis):
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{lbl}</div>
            <div class='metric-sub'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# WHAT'S INSIDE — the three story pages
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>What's in This Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>Three pages walk from the old method to the new model and its results</div>",
            unsafe_allow_html=True)

p1, p2, p3 = st.columns(3)
pages = [
    (p1, "#4A90C4", "1", "Overview",
     "The headline numbers, network and school predictions for next year, and data-coverage notes."),
    (p2, "#C8973A", "2", "Old System — CSR",
     "How the legacy Cohort-Survival-Rate method projects enrollment, and where it systematically misses."),
    (p3, "#22C55E", "3", "ML Model Intelligence",
     "The Gradient-Boosted-Tree methodology, what drives it, accuracy, and the head-to-head vs CSR."),
]
for col, color, num, title, body in pages:
    with col:
        st.markdown(f"""
        <div style='background:#FFFFFF; border:1px solid #E2E8F0; border-top:4px solid {color};
                    border-radius:12px; padding:20px 18px; height:100%;
                    box-shadow:0 1px 4px rgba(0,48,87,0.06);'>
            <div style='display:flex; align-items:center; gap:10px; margin-bottom:8px;'>
                <div style='min-width:30px; height:30px; background:{color}; border-radius:50%;
                            display:flex; align-items:center; justify-content:center;
                            font-weight:800; color:white;'>{num}</div>
                <div style='font-weight:800; font-size:1rem; color:#003057;'>{title}</div>
            </div>
            <div style='font-size:0.82rem; color:#4A6580; line-height:1.6;'>{body}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTIONS — network level + school level
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"<div class='section-header'>{FYEAR} Predictions — Network &amp; School Level</div>",
            unsafe_allow_html=True)
st.markdown("<div class='section-sub'>Grade-level projections summed to the level each leader plans at</div>",
            unsafe_allow_html=True)

# Network-wide single number
st.markdown(f"""
<div class='info-box'>
    <b>Entire network:</b> {network_latest:,} students enrolled in {LATEST} →
    <b>{network_forecast:,} projected for {FYEAR}</b>
    ({change_pct:+.1f}%, a change of {network_forecast - network_latest:+,} students).
</div>
""", unsafe_allow_html=True)

# Per-network (region) breakdown
net_tbl = (fc.groupby("REGION", as_index=False)
           .agg(**{f"{LATEST} Actual": ("ACTUAL_LATEST", "sum"),
                   f"{FYEAR} Forecast": ("FORECAST_ENROLLMENT", "sum")}))
net_tbl["Change"] = net_tbl[f"{FYEAR} Forecast"] - net_tbl[f"{LATEST} Actual"]
net_tbl["Change %"] = (net_tbl["Change"] / net_tbl[f"{LATEST} Actual"] * 100).round(1)
net_tbl = net_tbl.sort_values(f"{FYEAR} Forecast", ascending=False).rename(columns={"REGION": "Network"})

# Per-school breakdown
school_tbl = (fc.groupby(["SCHOOL_KEY", "SCHOOL_LABEL", "REGION"], as_index=False)
              .agg(**{f"{LATEST} Actual": ("ACTUAL_LATEST", "sum"),
                      f"{FYEAR} Forecast": ("FORECAST_ENROLLMENT", "sum")}))
school_tbl["Change"] = school_tbl[f"{FYEAR} Forecast"] - school_tbl[f"{LATEST} Actual"]
school_tbl["Change %"] = (school_tbl["Change"] /
                          school_tbl[f"{LATEST} Actual"].replace(0, pd.NA) * 100).round(1)
school_tbl = (school_tbl
              .drop(columns=["SCHOOL_KEY"])
              .rename(columns={"SCHOOL_LABEL": "School", "REGION": "Network"})
              .sort_values(f"{FYEAR} Forecast", ascending=False))

tab_net, tab_school = st.tabs([f"🏙️  By Network ({len(net_tbl)})",
                               f"🏫  By School ({len(school_tbl):,})"])

_intfmt = {f"{LATEST} Actual": "{:,.0f}", f"{FYEAR} Forecast": "{:,.0f}",
           "Change": "{:+,.0f}", "Change %": "{:+.1f}%"}
with tab_net:
    st.dataframe(net_tbl.style.format(_intfmt), hide_index=True, width="stretch")
with tab_school:
    st.dataframe(school_tbl.style.format(_intfmt), hide_index=True, width="stretch", height=420)
    st.download_button(
        "⬇️  Download school-level predictions (CSV)",
        school_tbl.to_csv(index=False).encode("utf-8"),
        file_name=f"enrollment_school_predictions_{FYEAR}.csv",
        mime="text/csv",
    )

st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# COVERAGE & DATA NOTES — answers the K1/K9 and missing-lag questions
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>Coverage &amp; Data Notes</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>What the model does and does not currently cover — read before "
            "quoting grade-level numbers</div>", unsafe_allow_html=True)

grades_present = sorted(int(g) for g in fc["GRADE"].unique())
csr_nulls = int(df["COHORT_SURVIVAL_RATE"].isna().sum()) if df["COHORT_SURVIVAL_RATE"].isna().any() else 0
# entry-grade rows (no in-system feeder) in the forecast set
entry_rows = int((fc["HAS_FEEDER_GRADE"] == 0).sum())

notes = [
    ("#EF4444", "K1 — Kindergarten is not in the data",
     f"The model's grade range is {grades_present[0]}–{grades_present[-1]}; Kindergarten (and Pre-K) "
     "are absent from the source table, so <b>no K1 forecast is produced today</b>. Kindergarten is an "
     "entry grade with no in-system feeder — projecting it would need a births-based driver (~5-year lag), "
     "not the grade-progression signal the rest of the model relies on."),
    ("#EF4444", "K9 — Grade 9 is missing (the high-school gap)",
     "Grades jump from 8 to 10 — <b>Grade 9 is not present</b> in the dataset. Because of that there is no "
     "Grade-9 projection, and Grade-10's feeder grade (which should be last year's Grade 9) has nothing to "
     "point at, so its cross-grade signal falls back to imputed values (see below)."),
    ("#C8973A", "Missing lag features are flagged, not faked",
     f"About {entry_rows:,} forecast rows are entry grades with no feeder, and cohort-survival-rate is "
     f"missing on {csr_nulls:,} historical rows. These gaps are <b>filled with train-only medians</b> and "
     "marked with the <code>HAS_FEEDER_GRADE</code> flag so the model learns to treat them as "
     "&ldquo;no history available&rdquo; rather than a real zero — never forward-filled from the value "
     "being predicted (which would leak)."),
]
for color, title, body in notes:
    st.markdown(f"""
    <div style='background:#FFFFFF; border:1px solid #E2E8F0; border-left:5px solid {color};
                border-radius:0 10px 10px 0; padding:16px 18px; margin-bottom:12px;
                box-shadow:0 1px 4px rgba(0,48,87,0.06);'>
        <div style='font-size:0.9rem; font-weight:700; color:#003057; margin-bottom:6px;'>{title}</div>
        <div style='font-size:0.82rem; color:#334D66; line-height:1.6;'>{body}</div>
    </div>
    """, unsafe_allow_html=True)

# Headline CSR-vs-ML teaser
wape = BACKTEST_HEADLINE["Budget error rate (WAPE)"]
st.markdown(f"""
<div class='insight-card'>
    <div class='title'>📌 Why the new model matters</div>
    <div class='body'>On a 4-year out-of-sample backtest the budget error rate (WAPE) falls from
    <b>{wape[0]}</b> under the legacy CSR method to <b>{wape[1]}</b> with the ML model. The
    <b>Old System — CSR</b> and <b>ML Model Intelligence</b> pages break this down in full.</div>
</div>
""", unsafe_allow_html=True)
