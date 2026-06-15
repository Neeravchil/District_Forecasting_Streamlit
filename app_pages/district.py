import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from utils.loader import load_data, build_forecast, district_year_totals, latest_year
from utils.forecast import (OFFICIAL_FI, BACKTEST_HEADLINE, MODEL_MAE,
                            MODEL_RMSE, MODEL_R2)
import pandas as pd

NETWORK_PALETTE = (px.colors.qualitative.Prism + px.colors.qualitative.Safe
                   + px.colors.qualitative.Vivid)

_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Aptos, Nunito Sans, Segoe UI, Arial", size=12, color="#334D66"),
    margin=dict(l=44, r=44, t=20, b=40),
    showlegend=True,
    legend=dict(orientation="h", y=1.08),
)

# Friendly names for the model's feature columns (district feature-importance chart)
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
    "SCHOOL_SIZE":                        "School size",
    "REGION_ENCODED":                     "Local area",
    "GOVERNANCE_ENCODED":                 "Governance type",
    "IS_HIGH_SCHOOL":                     "Is high school",
    "IS_SELECTIVE":                       "Is selective",
    "IS_ATTENDANCE_AREA":                 "Is attendance-area",
    "IS_SMALL_SCHOOL":                    "Is small school",
    "HAS_FEEDER_GRADE":                   "Has feeder grade",
    "IS_MIGRANT_ANOMALY_YEAR":            "Migrant-anomaly year",
}

# ── Load data & forecast ──────────────────────────────────────────────────────
df = load_data()
fc = build_forecast(df)
yr_totals = district_year_totals(df)
LATEST = latest_year(df)
FYEAR = LATEST + 1

latest_total   = int(df[df["SCHOOL_YEAR"] == LATEST]["ENROLLMENT"].sum())
forecast_total = int(fc["FORECAST_ENROLLMENT"].sum())
n_schools      = int(df[(df["SCHOOL_YEAR"] == LATEST) & (df["IS_SCHOOL_OPEN"] == 1)]["SCHOOL_KEY"].nunique())
change_pct     = (forecast_total - latest_total) / latest_total * 100

# ── Sidebar filters (scatter only) ───────────────────────────────────────────
all_networks = sorted(n for n in fc["NETWORK"].dropna().unique() if n != "Unassigned")
with st.sidebar:
    st.markdown("<p style='font-size:0.82rem; font-weight:700; color:#E8EDF2; "
                "margin:6px 0 4px 0; letter-spacing:0.03em;'>Scatter filter</p>",
                unsafe_allow_html=True)
    network_filter = st.multiselect(
        "Show networks", options=all_networks, default=all_networks,
    )
    size_min = st.slider("Min school enrollment", min_value=0, max_value=2000,
                         value=0, step=25)

# ══════════════════════════════════════════════════════════════════════════════
# HERO BANNER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='background:linear-gradient(135deg,#003057 0%,#00497a 100%);
            border-radius:12px; padding:40px 48px; margin-bottom:28px;
            border-left:6px solid #C8973A; width:100%;'>
    <div style='font-size:2.2rem; font-weight:800; color:#FFFFFF; line-height:1.2; margin-bottom:16px;'>
        Projecting Next Year's Enrollment Before Budgets Are Set
    </div>
    <div style='font-size:0.97rem; color:#B8CFDF; line-height:1.7;'>
        District 299 builds its staffing and funding plan around how many students will walk
        through the door each fall. This tool uses a machine-learning model trained on
        <b style='color:#FFFFFF;'>seven years of school × grade records</b> to project
        {FYEAR} enrollment for every open school — so leaders can plan with a number,
        not a guess.
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI strip ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
kpis = [
    (f"{latest_total:,}",   f"{LATEST} Enrollment",          "Latest actual district total"),
    (f"{forecast_total:,}", f"{FYEAR} Projected Enrollment", "Model forecast, all open schools"),
    (f"{change_pct:+.1f}%", "Projected Change",              f"{FYEAR} vs {LATEST}"),
    (f"{n_schools:,}",      "Open Schools",                  f"Forecast at the school × grade level"),
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
# WHY IT MATTERS  /  HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════
col_l, col_r = st.columns([1.05, 1], gap="large")

with col_l:
    st.markdown("""
    <div style='font-size:1.25rem; font-weight:800; color:#003057; margin-bottom:4px;
                border-left:5px solid #C8973A; padding-left:14px;'>
        Why Enrollment Projections Matter
    </div>
    <div style='font-size:0.82rem; color:#4A6580; margin-bottom:18px; padding-left:19px;'>
        What the numbers drive and why an accurate forecast is essential
    </div>
    """, unsafe_allow_html=True)

    reasons = [
        ("#4A90C4", "💵", "Enrollment is the engine of the budget",
         "Per-pupil funding means a school's entire staffing allocation rides on its projected headcount. A forecast that is off by 30 students is a forecast that is off by a teacher."),
        ("#003057", "🪑", "Seats, sections, and classrooms",
         "Grade-level projections decide how many sections to open, how many rooms to assign, and where space is tight or going empty — months before the first bell."),
        ("#C8973A", "📉", "Decline is uneven, not uniform",
         "District enrollment has trended down, but the drop is concentrated in specific grades, schools, and networks. A single district number hides where the real pressure is."),
        ("#22C55E", "🎯", "Early enough to act",
         "Projecting the coming year now gives leadership a full planning cycle to rebalance staff, consolidate sections, and target retention before the year locks in."),
    ]
    for color, icon, title, body in reasons:
        st.markdown(f"""
        <div style='background:#FFFFFF; border:1px solid #E2E8F0; border-left:5px solid {color};
                    border-radius:0 10px 10px 0; padding:16px 18px; margin-bottom:12px;
                    box-shadow:0 1px 4px rgba(0,48,87,0.06);'>
            <div style='font-size:0.9rem; font-weight:700; color:#003057; margin-bottom:6px;'>
                {icon}&nbsp; {title}
            </div>
            <div style='font-size:0.82rem; color:#334D66; line-height:1.6;'>{body}</div>
        </div>
        """, unsafe_allow_html=True)

with col_r:
    st.markdown("""
    <div style='font-size:1.25rem; font-weight:800; color:#003057; margin-bottom:4px;
                border-left:5px solid #C8973A; padding-left:14px;'>
        From School Records to a Projection — How It Works
    </div>
    <div style='font-size:0.82rem; color:#4A6580; margin-bottom:18px; padding-left:19px;'>
        Five steps from raw records to a school × grade projection
    </div>
    """, unsafe_allow_html=True)

    stages = [
        ("#4A90C4", "1", "Aggregate seven years of records",
         "One row per school × grade × year — counting distinct students from the annualised enrollment table across the whole district."),
        ("#8B5CF6", "2", "Engineer leak-free lag features",
         "Last year's count, the feeder grade's count, cohort survival rates, and school totals — every feature uses only prior-year data, never the value being predicted."),
        ("#C8973A", "3", "Train a Gradient Boosted Tree ensemble",
         "One hundred decision trees are trained in PySpark MLlib, with the 2022–24 migrant-surge years down-weighted so they inform without dominating the model."),
        ("#EF4444", "4", "Score next year without Spark",
         "The trained trees are stored as Parquet and scored in pure Python — no Java runtime needed. Projections run in milliseconds on any machine."),
        ("#22C55E", "5", "Roll up to school, network, or district",
         "Grade-level projections sum cleanly to any level, so every leader sees a forecast scoped to their span of control."),
    ]
    for color, num, title, desc in stages:
        st.markdown(f"""
        <div style='display:flex; align-items:flex-start; margin-bottom:14px; gap:14px;
                    background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px;
                    padding:14px 16px; box-shadow:0 1px 4px rgba(0,48,87,0.06);'>
            <div style='min-width:36px; height:36px; background:{color}; border-radius:50%;
                        display:flex; align-items:center; justify-content:center;
                        font-weight:800; font-size:1rem; color:white; flex-shrink:0;'>{num}</div>
            <div>
                <div style='font-weight:700; font-size:0.92rem; color:#003057; margin-bottom:4px;'>{title}</div>
                <div style='font-size:0.8rem; color:#4A6580; line-height:1.5;'>{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DISTRICT ENROLLMENT TREND + NEXT-YEAR FORECAST
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>District-Wide Enrollment Trend &amp; Next-Year Forecast</div>",
            unsafe_allow_html=True)
st.markdown(
    f"<div class='section-sub'>Actual totals {int(yr_totals['SCHOOL_YEAR'].min())}–{LATEST} "
    f"(open schools) · {FYEAR} is the model projection · Shaded = ±RMSE band scaled by school count</div>",
    unsafe_allow_html=True,
)

hist_years = yr_totals["SCHOOL_YEAR"].astype(int).tolist()
hist_vals  = yr_totals["ENROLLMENT"].tolist()
x_labels   = [str(y) for y in hist_years] + [f"{FYEAR}"]

# Band: per-row RMSE summed in quadrature (independent errors) across school-grades
n_rows = len(fc)
band = MODEL_RMSE * (n_rows ** 0.5)
fc_upper = forecast_total + band
fc_lower = forecast_total - band

fig_fc = go.Figure()
fig_fc.add_trace(go.Bar(
    name="Actual enrollment", x=[str(y) for y in hist_years], y=hist_vals,
    marker_color="#003057",
))
fig_fc.add_trace(go.Bar(
    name=f"{FYEAR} projection", x=[f"{FYEAR}"], y=[forecast_total],
    marker_color="rgba(200,151,58,0.55)",
    marker_line=dict(color="#C8973A", width=1.5),
    marker_pattern_shape="/", marker_pattern_fgcolor="#C8973A",
))
# Trend line connecting last actual -> forecast
fig_fc.add_trace(go.Scatter(
    name="Projection trend",
    x=[str(hist_years[-1]), f"{FYEAR}"], y=[hist_vals[-1], forecast_total],
    mode="lines+markers",
    line=dict(color="#C8973A", width=2.5, dash="dash"),
    marker=dict(size=9, color="#C8973A", symbol="circle-open"),
))
fig_fc.add_trace(go.Scatter(
    x=[f"{FYEAR}", f"{FYEAR}"], y=[fc_lower, fc_upper],
    mode="lines", line=dict(color="#C8973A", width=10),
    opacity=0.18, showlegend=False, name="_band",
))
fig_fc.add_annotation(
    x=f"{FYEAR}", y=forecast_total, text=f"  {forecast_total:,}",
    showarrow=False, font=dict(size=12, color="#8a6a1e", weight=700),
    xanchor="left", yanchor="middle",
)
fig_fc.update_layout(
    **_LAYOUT, barmode="group",
    xaxis=dict(title="School year", categoryorder="array", categoryarray=x_labels,
               showgrid=False),
    yaxis=dict(title="Total enrollment", showgrid=True, gridcolor="#F1F5F9",
               tickformat=","),
    height=440,
)
st.plotly_chart(fig_fc, use_container_width=True)

st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# WHAT DRIVES THE FORECAST — feature importance
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>What Drives the Forecast?</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>Spark GBT feature importance (sums to 1.0) — "
            "from the model review document, Section 4</div>",
            unsafe_allow_html=True)

fi = (pd.DataFrame({"feature": list(OFFICIAL_FI.keys()),
                    "importance": list(OFFICIAL_FI.values())})
      .head(10).copy())
fi["label"] = fi["feature"].map(FEATURE_NAMES).fillna(fi["feature"])
fi["pct"] = fi["importance"] * 100

fig_fi = go.Figure(go.Bar(
    x=fi["pct"][::-1], y=fi["label"][::-1], orientation="h",
    marker_color="#4A90C4",
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
st.plotly_chart(fig_fi, use_container_width=True)

st.markdown("""
<div class='insight-card'>
    <div class='title'>📌 Same-grade history dominates (32%)</div>
    <div class='body'>If a school had 80 sixth-graders last year, the strongest single predictor of its
    seventh-graders is exactly that number. Context features — the district-wide grade trend (14%),
    school total (12%), and the grade itself (11%) — add another ~37%. The cross-grade promotion
    signals (feeder lags + survival rates, ~19% combined) mean the model still uses the CSR insight,
    but as one input among many rather than the whole forecast.</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL ACCURACY
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>How Accurate Is the Model?</div>", unsafe_allow_html=True)
st.markdown(f"<div class='section-sub'>Single-fold validation — trained on 2020–{LATEST - 1}, "
            f"tested on the held-out SY{LATEST} — per school × grade enrollment</div>",
            unsafe_allow_html=True)

acc1, acc2, acc3, acc4 = st.columns(4)
accuracy_kpis = [
    ("#4A90C4", f"{MODEL_MAE:.1f}",   "MAE (students)",
     "Mean absolute error — the typical school-grade projection lands within ~8 students of actual."),
    ("#003057", f"{MODEL_RMSE:.1f}",  "RMSE (students)",
     "Root mean squared error — larger than MAE because a few big schools carry most of the error mass."),
    ("#C8973A", f"{MODEL_R2:.2f}",    "R²",
     f"The model explains {MODEL_R2:.0%} of the variance in school-grade enrollment on the unseen year."),
    ("#8B5CF6", "100",                "Boosted trees",
     "Depth-5 trees, learning rate 0.1, subsampling 0.8 — migrant-surge years down-weighted to 0.3."),
]
for col, (color, val, lbl, body) in zip([acc1, acc2, acc3, acc4], accuracy_kpis):
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
# CSR vs ML — leadership headline (review PDF, Sections 1–2)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>ML vs the Current CSR Method</div>",
            unsafe_allow_html=True)
st.markdown("<div class='section-sub'>4-year out-of-sample walk-forward backtest, SY2023–2026 — "
            "18,899 school-grade predictions across 842 open schools (model review document)</div>",
            unsafe_allow_html=True)

hl = pd.DataFrame(
    [(metric, csr, ml) for metric, (csr, ml) in BACKTEST_HEADLINE.items()],
    columns=["Headline metric (4-year backtest, district-wide)",
             "CSR (current SAS)", "ML (proposed GBT)"],
)
st.dataframe(hl, hide_index=True, use_container_width=True)

st.markdown("""
<div class='insight-card'>
    <div class='title'>📌 A ~4× reduction in district-wide forecast bias</div>
    <div class='body'>The current SAS Cohort Survival Rate method over-predicted enrollment by
    <b>+193,048 students (+27.3%)</b> across the backtest; the GBT model cuts that to
    <b>−51,357 (−7.3%)</b> — slightly conservative instead of heavily over-budgeted. The budget
    error rate (WAPE) falls from 71.4% to 24.7%, and the model wins on MAE in 21 of 23 networks,
    in every backtest year, and in 78.9% of individual schools. At ~$6,200 per pupil
    (FY26 Student-Based Budgeting base rate), that bias gap is the headline dollar number
    for leadership.</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SCHOOL-LEVEL SCATTER — current size vs projected change
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>Where Enrollment Is Growing or Shrinking</div>",
            unsafe_allow_html=True)
st.markdown(f"<div class='section-sub'>Each bubble is one school — {LATEST} enrollment vs projected "
            f"% change into {FYEAR}, coloured by network</div>", unsafe_allow_html=True)

school_fc = (fc.groupby(["SCHOOL_KEY", "SCHOOL_LABEL", "NETWORK"], as_index=False)
             .agg(LATEST_ENR=("ACTUAL_LATEST", "sum"),
                  FORECAST_ENR=("FORECAST_ENROLLMENT", "sum")))
school_fc = school_fc[school_fc["LATEST_ENR"] > 0].copy()
school_fc["change_pct"] = ((school_fc["FORECAST_ENR"] - school_fc["LATEST_ENR"])
                           / school_fc["LATEST_ENR"] * 100).round(1)
school_fc["change_pct"] = school_fc["change_pct"].clip(-50, 50)

scatter_df = school_fc[
    (school_fc["NETWORK"].isin(network_filter)) &
    (school_fc["LATEST_ENR"] >= size_min)
].copy()

fig2 = px.scatter(
    scatter_df, x="LATEST_ENR", y="change_pct",
    size="LATEST_ENR", color="NETWORK",
    color_discrete_sequence=NETWORK_PALETTE, size_max=26,
    hover_data={"SCHOOL_LABEL": True, "LATEST_ENR": ":,", "FORECAST_ENR": ":,",
                "change_pct": ":.1f", "NETWORK": True},
    labels={"LATEST_ENR": f"{LATEST} enrollment", "change_pct": f"Projected % change → {FYEAR}",
            "FORECAST_ENR": f"{FYEAR} projection", "SCHOOL_LABEL": "School", "NETWORK": "Network"},
)
fig2.add_hline(y=0, line_dash="dot", line_color="#64748B", line_width=1.5,
               annotation_text="No change", annotation_position="top right",
               annotation_font_color="#64748B", annotation_font_size=11)
_scatter_layout = {**_LAYOUT, "legend": dict(orientation="v", y=1, x=1.02, font=dict(size=9))}
fig2.update_layout(
    **_scatter_layout,
    xaxis=dict(title=f"{LATEST} enrollment", showgrid=True, gridcolor="#F1F5F9"),
    yaxis=dict(title=f"Projected % change → {FYEAR}", showgrid=True, gridcolor="#F1F5F9"),
    height=520,
)
st.plotly_chart(fig2, use_container_width=True)

if scatter_df.empty:
    st.warning("No schools match the current filters.")
