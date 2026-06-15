"""
Page 3 — ML Model Intelligence.

Showcases the new Gradient-Boosted-Tree methodology: how it is built, what
drives it, how accurate it is, and the head-to-head against the legacy CSR
method over the 4-year backtest.
"""

import pandas as pd
import plotly.graph_objects as go

import streamlit as st

from utils.forecast import (OFFICIAL_FI, BACKTEST_HEADLINE, BACKTEST_BY_YEAR,
                            MODEL_MAE, MODEL_RMSE, MODEL_R2)

FEATURE_NAMES = {
    "SAME_GRADE_LAST_YEAR":               "Same grade — last year",
    "DISTRICT_GRADE_ENROLLMENT_LAST_YEAR":"District grade total — last year",
    "SCHOOL_TOTAL_LAST_YEAR":             "School total — last year",
    "GRADE_idx":                          "Grade level",
    "GRADE_NUMERIC":                      "Grade (numeric order)",
    "FEEDER_GRADE_LAST_YEAR":             "Feeder grade — last year",
    "FEEDER_GRADE_2YR_AGO":               "Feeder grade — 2 years ago",
    "COHORT_SURVIVAL_RATE":               "Cohort survival rate",
    "AVG_SURVIVAL_RATE_3YR":              "Avg survival rate (3 yr)",
    "SCHOOL_EFFECT":                      "School effect (history)",
    "SAME_GRADE_2YR_AGO":                 "Same grade — 2 years ago",
    "REGION_ENCODED":                     "Region",
    "GOVERNANCE_ENCODED":                 "Governance type",
    "IS_HIGH_SCHOOL":                     "Is high school",
    "IS_SELECTIVE":                       "Is selective",
    "IS_ATTENDANCE_AREA":                 "Is attendance-area",
    "IS_SMALL_SCHOOL":                    "Is small school",
    "HAS_FEEDER_GRADE":                   "Has feeder grade",
    "IS_MIGRANT_ANOMALY_YEAR":            "Migrant-anomaly year",
}

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='background:linear-gradient(135deg,#053a2c 0%,#0a5a43 100%);
            border-radius:12px; padding:40px 48px; margin-bottom:28px;
            border-left:6px solid #22C55E; width:100%;'>
    <div style='font-size:0.8rem; font-weight:700; color:#9BE8C4; letter-spacing:0.12em;
                text-transform:uppercase; margin-bottom:10px;'>The New Model</div>
    <div style='font-size:2.2rem; font-weight:800; color:#FFFFFF; line-height:1.2; margin-bottom:16px;'>
        ML Model Intelligence
    </div>
    <div style='font-size:0.97rem; color:#C9EAD9; line-height:1.7;'>
        A <b style='color:#FFFFFF;'>Gradient-Boosted-Tree</b> ensemble trained in PySpark MLlib.
        It keeps the cohort-survival insight but treats it as one input among many — learning from
        100 decision trees how school size, grade, region, history, and survival signals combine to
        predict next year's enrollment.
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HOW IT'S BUILT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>How the Model Is Built</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>From seven years of records to a school × grade projection</div>",
            unsafe_allow_html=True)

stages = [
    ("#4A90C4", "1", "Aggregate 7 years of records",
     "One row per school × grade × year, counting distinct students across the whole network."),
    ("#8B5CF6", "2", "Engineer leak-free lag features",
     "Last year's count, feeder-grade counts, cohort-survival rates, school totals — every feature uses only prior-year data."),
    ("#C8973A", "3", "Train 100 boosted trees",
     "Depth-5 trees, learning rate 0.1, subsampling 0.8 — the 2022–24 migrant-surge years down-weighted to 0.3 so they inform without dominating."),
    ("#EF4444", "4", "Score next year without Spark",
     "Trees are stored as Parquet and scored in pure Python — projections run in milliseconds, no Java runtime needed."),
    ("#22C55E", "5", "Roll up to any level",
     "Grade-level projections sum cleanly to school, network, or district."),
]
scols = st.columns(5)
for col, (color, num, title, desc) in zip(scols, stages):
    with col:
        st.markdown(f"""
        <div style='background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:14px;
                    height:100%; box-shadow:0 1px 4px rgba(0,48,87,0.06);'>
            <div style='min-width:30px; height:30px; background:{color}; border-radius:50%;
                        display:flex; align-items:center; justify-content:center;
                        font-weight:800; color:white; margin-bottom:8px;'>{num}</div>
            <div style='font-weight:700; font-size:0.8rem; color:#003057; margin-bottom:5px;'>{title}</div>
            <div style='font-size:0.74rem; color:#4A6580; line-height:1.45;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# WHAT DRIVES IT — feature importance
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>What Drives the Forecast?</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>Spark GBT feature importance (sums to 1.0) — top 10 of 19 features</div>",
            unsafe_allow_html=True)

fi = (pd.DataFrame({"feature": list(OFFICIAL_FI.keys()),
                    "importance": list(OFFICIAL_FI.values())})
      .head(10).copy())
fi["label"] = fi["feature"].map(FEATURE_NAMES).fillna(fi["feature"])
fi["pct"] = fi["importance"] * 100

fig_fi = go.Figure(go.Bar(
    x=fi["pct"][::-1], y=fi["label"][::-1], orientation="h",
    marker_color="#22A36B",
    text=[f"{v:.1f}%" for v in fi["pct"][::-1]],
    textposition="outside", textfont=dict(size=10, color="#334D66"),
))
fig_fi.update_layout(
    paper_bgcolor="white", plot_bgcolor="white",
    font=dict(family="Aptos, Nunito Sans, Segoe UI, Arial", size=12, color="#334D66"),
    margin=dict(l=10, r=70, t=10, b=10), showlegend=False,
    xaxis=dict(showgrid=True, gridcolor="#F1F5F9", showticklabels=False,
               range=[0, fi["pct"].max() * 1.25]),
    yaxis=dict(showgrid=False, tickfont=dict(size=11)),
    height=380,
)
st.plotly_chart(fig_fi, width="stretch")

st.markdown("""
<div class='insight-card'>
    <div class='title'>📌 History dominates, but context fills the gap</div>
    <div class='body'>The single strongest signal is the same grade's count last year (32%). What makes
    the model better than CSR is the next ~37% — the district-wide grade trend, school size, and grade
    level — context CSR never sees. The cohort-survival signals (feeder lags + survival rates, ~19%
    combined) are still used, but as inputs among many rather than the entire forecast.</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ACCURACY
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>How Accurate Is It?</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>Single-fold validation — trained 2020–2025, tested on the held-out "
            "SY2026 — per school × grade</div>", unsafe_allow_html=True)

acc = st.columns(4)
accuracy_kpis = [
    ("#22A36B", f"{MODEL_MAE:.1f}", "MAE (students)",
     "The typical school-grade projection lands within ~8 students of actual."),
    ("#003057", f"{MODEL_RMSE:.1f}", "RMSE (students)",
     "Larger than MAE because a few big schools carry most of the error mass."),
    ("#C8973A", f"{MODEL_R2:.2f}", "R²",
     f"The model explains {MODEL_R2:.0%} of the variance on the unseen year."),
    ("#8B5CF6", "100", "Boosted trees",
     "Depth-5, learning rate 0.1, subsampling 0.8 — migrant years down-weighted to 0.3."),
]
for col, (color, val, lbl, body) in zip(acc, accuracy_kpis):
    with col:
        st.markdown(f"""
        <div style='background:#FFFFFF; border:1px solid #D1DBE8; border-top:4px solid {color};
                    border-radius:12px; padding:20px 18px; height:100%;
                    box-shadow:0 1px 4px rgba(0,48,87,0.06);'>
            <div style='font-size:2rem; font-weight:800; color:{color};'>{val}</div>
            <div style='font-size:0.78rem; font-weight:700; color:#1E293B;
                        text-transform:uppercase; letter-spacing:0.04em; margin-top:6px;'>{lbl}</div>
            <div style='font-size:0.76rem; color:#64748B; margin-top:6px; line-height:1.5;'>{body}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HEAD TO HEAD — CSR vs ML by year
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>ML vs CSR — Year by Year</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>Average error per school-grade (MAE, lower is better) across the "
            "4-year walk-forward backtest</div>", unsafe_allow_html=True)

years   = [str(r[0]) for r in BACKTEST_BY_YEAR]
mae_csr = [r[2] for r in BACKTEST_BY_YEAR]
mae_ml  = [r[3] for r in BACKTEST_BY_YEAR]

fig = go.Figure()
fig.add_trace(go.Bar(name="CSR (legacy)", x=years, y=mae_csr, marker_color="#C8973A",
                     text=[f"{v:.1f}" for v in mae_csr], textposition="outside"))
fig.add_trace(go.Bar(name="ML (new)", x=years, y=mae_ml, marker_color="#22A36B",
                     text=[f"{v:.1f}" for v in mae_ml], textposition="outside"))
fig.update_layout(
    barmode="group", paper_bgcolor="white", plot_bgcolor="white",
    font=dict(family="Aptos, Nunito Sans, Segoe UI, Arial", size=12, color="#334D66"),
    margin=dict(l=44, r=44, t=20, b=40),
    legend=dict(orientation="h", y=1.12),
    xaxis=dict(title="School year", showgrid=False),
    yaxis=dict(title="Error per school-grade (MAE)", showgrid=True, gridcolor="#F1F5F9"),
    height=420,
)
st.plotly_chart(fig, width="stretch")

hl = pd.DataFrame(
    [(metric, csr, ml) for metric, (csr, ml) in BACKTEST_HEADLINE.items()],
    columns=["Headline metric (4-year backtest, district-wide)", "CSR (legacy)", "ML (new)"],
)
st.dataframe(hl, hide_index=True, width="stretch")

st.markdown(f"""
<div class='insight-card'>
    <div class='title'>📌 A ~4× cut in average error, and the bias flips</div>
    <div class='body'>The ML model lowers the average error per school-grade from
    <b>{BACKTEST_HEADLINE['Average error per school-grade (MAE)'][0]}</b> to
    <b>{BACKTEST_HEADLINE['Average error per school-grade (MAE)'][1]}</b> students, and turns CSR's
    <b>{BACKTEST_HEADLINE['as % of actual enrollment'][0]}</b> over-prediction into a slight
    <b>{BACKTEST_HEADLINE['as % of actual enrollment'][1]}</b> — conservative instead of heavily
    over-budgeted. R² moves from {BACKTEST_HEADLINE['Variance explained (R²)'][0]} to
    {BACKTEST_HEADLINE['Variance explained (R²)'][1]}.</div>
</div>
""", unsafe_allow_html=True)
