"""
Page 1 — Overview.

Plain-language summary of the enrollment forecast for the planning meeting:
what it does, next year's numbers for the whole network and for each school,
and the data-coverage notes that answer the Kindergarten / Grade-9 and
missing-history questions.
"""

import pandas as pd
import streamlit as st

from utils.loader import load_data, build_forecast, latest_year
from utils.forecast import MODEL_MAE, BACKTEST_HEADLINE

# ── Data & forecast ───────────────────────────────────────────────────────────
df = load_data()
fc = build_forecast(df)
LATEST = latest_year(df)
FYEAR = LATEST + 1

network_latest   = int(df[df["SCHOOL_YEAR"] == LATEST]["ENROLLMENT"].sum())
network_forecast = int(fc["FORECAST_ENROLLMENT"].sum())
change_pct       = (network_forecast - network_latest) / network_latest * 100
n_schools        = int(fc["SCHOOL_KEY"].nunique())
n_regions        = int(fc["REGION"].nunique())

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
        Our Enrollment Forecast for {FYEAR}
    </div>
    <div style='font-size:0.97rem; color:#B8CFDF; line-height:1.7;'>
        This dashboard shows where our new enrollment forecast stands for the {FYEAR} planning year.
        It replaces the old Cohort Survival Rate (CSR) spreadsheet with a model that
        <b style='color:#FFFFFF;'>learns from seven years of real enrollment</b> — and gives a number
        for the <b style='color:#FFFFFF;'>whole network</b> as well as for
        <b style='color:#FFFFFF;'>every single school</b>.
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI strip ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
kpis = [
    (f"{network_forecast:,}", f"{FYEAR} Forecast",   "Students expected across the whole network"),
    (f"{change_pct:+.1f}%",   "Expected Change",     f"Compared with {LATEST}"),
    (f"{n_schools:,}",        "Schools Covered",     "Every open school, every grade"),
    (f"±{MODEL_MAE:.0f}",     "Typical Accuracy",    "How close each school-grade estimate usually lands"),
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
st.markdown("<div class='section-sub'>Three pages: where we are, the old way of forecasting, "
            "and the new one</div>", unsafe_allow_html=True)

p1, p2, p3 = st.columns(3)
pages = [
    (p1, "#4A90C4", "1", "Overview",
     "The headline numbers, next year's predictions for the network and for each school, and what the data does and doesn't cover."),
    (p2, "#C8973A", "2", "The Old Way — CSR",
     "How our long-standing Cohort Survival Rate method forecasts enrollment, and where it tends to miss."),
    (p3, "#22C55E", "3", "The New Model",
     "How the new model works, what information it leans on most, and how much more accurate it is."),
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
# PREDICTIONS — whole network + by region + by school
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"<div class='section-header'>{FYEAR} Predictions</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>The forecast for the whole network, and broken down the way "
            "leaders plan — by region and by school</div>", unsafe_allow_html=True)

# Whole-network headline
st.markdown(f"""
<div class='info-box'>
    <b>Whole network:</b> {network_latest:,} students enrolled in {LATEST} →
    <b>{network_forecast:,} expected in {FYEAR}</b>
    ({change_pct:+.1f}%, a change of {network_forecast - network_latest:+,} students).
</div>
""", unsafe_allow_html=True)

# By-region breakdown
region_tbl = (fc.groupby("REGION", as_index=False)
              .agg(**{f"{LATEST} Actual": ("ACTUAL_LATEST", "sum"),
                      f"{FYEAR} Forecast": ("FORECAST_ENROLLMENT", "sum")}))
region_tbl["Change"] = region_tbl[f"{FYEAR} Forecast"] - region_tbl[f"{LATEST} Actual"]
region_tbl["Change %"] = (region_tbl["Change"] / region_tbl[f"{LATEST} Actual"] * 100).round(1)
region_tbl = region_tbl.sort_values(f"{FYEAR} Forecast", ascending=False).rename(columns={"REGION": "Region"})

# By-school breakdown
school_tbl = (fc.groupby(["SCHOOL_KEY", "SCHOOL_LABEL", "REGION"], as_index=False)
              .agg(**{f"{LATEST} Actual": ("ACTUAL_LATEST", "sum"),
                      f"{FYEAR} Forecast": ("FORECAST_ENROLLMENT", "sum")}))
school_tbl["Change"] = school_tbl[f"{FYEAR} Forecast"] - school_tbl[f"{LATEST} Actual"]
school_tbl["Change %"] = (school_tbl["Change"] /
                          school_tbl[f"{LATEST} Actual"].replace(0, pd.NA) * 100).round(1)
school_tbl = (school_tbl
              .drop(columns=["SCHOOL_KEY"])
              .rename(columns={"SCHOOL_LABEL": "School", "REGION": "Region"})
              .sort_values(f"{FYEAR} Forecast", ascending=False))

tab_region, tab_school = st.tabs([f"🗺️  By Region ({len(region_tbl)})",
                                  f"🏫  By School ({len(school_tbl):,})"])

_fmt = {f"{LATEST} Actual": "{:,.0f}", f"{FYEAR} Forecast": "{:,.0f}",
        "Change": "{:+,.0f}", "Change %": "{:+.1f}%"}
with tab_region:
    st.dataframe(region_tbl.style.format(_fmt, na_rep="—"), hide_index=True, width="stretch")
with tab_school:
    st.dataframe(school_tbl.style.format(_fmt, na_rep="—"), hide_index=True,
                 width="stretch", height=420)
    st.download_button(
        "⬇️  Download school-by-school predictions (CSV)",
        school_tbl.to_csv(index=False).encode("utf-8"),
        file_name=f"enrollment_school_predictions_{FYEAR}.csv",
        mime="text/csv",
    )

st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# COVERAGE NOTES — answers the Kindergarten / Grade-9 / missing-history questions
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>What the Forecast Covers — and What It Doesn't</div>",
            unsafe_allow_html=True)
st.markdown("<div class='section-sub'>Worth knowing before quoting any single grade's number</div>",
            unsafe_allow_html=True)

grades_present = sorted(int(g) for g in fc["GRADE"].unique())

notes = [
    ("#EF4444", "Kindergarten isn't in the data yet",
     f"The forecast covers grades {grades_present[0]}–{grades_present[-1]}. Kindergarten (and Pre-K) "
     "simply aren't in our records, so there's <b>no Kindergarten forecast today</b>. Kindergarten is "
     "also special: it has no earlier grade inside the school to learn from — it depends mostly on how "
     "many children were born in the area about five years earlier — so it needs its own data source "
     "before we can project it."),
    ("#EF4444", "Grade 9 is missing too",
     "Our records jump straight from Grade 8 to Grade 10, so <b>there's no Grade 9 to report on</b>. "
     "That gap also means Grade 10 can't look back at last year's Grade 9, so we fill that one spot with "
     "a sensible average (see the next note)."),
    ("#C8973A", "Where there's no history, we fill carefully",
     "Some grades have no earlier-year number to look back on — the lowest grade in a school, or the "
     "missing Grade 9. In those spots we fill in a typical value from past years and clearly mark it as "
     "<b>&ldquo;no history available&rdquo;</b>, so the forecast treats it as unknown rather than as "
     "zero students. We never peek at the very number we're trying to predict."),
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

# Headline teaser
wape = BACKTEST_HEADLINE["Budget error rate (WAPE)"]
st.markdown(f"""
<div class='insight-card'>
    <div class='title'>📌 Why the new model matters</div>
    <div class='body'>Tested across four years it had never seen before, the new model's budgeting error
    dropped from <b>{wape[0]}</b> with the old CSR method to <b>{wape[1]}</b>. The next two pages show
    how the old and new approaches work, side by side.</div>
</div>
""", unsafe_allow_html=True)
