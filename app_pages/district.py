import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from utils.loader import (load_data, build_forecast, district_year_totals,
                          latest_year, with_network)
from utils.forecast import MODEL_RMSE  # forecast ±RMSE confidence band only
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

# ── Load data & forecast ──────────────────────────────────────────────────────
df = load_data()
fc = with_network(build_forecast(df), df)
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

# ── Pointer: model detail lives on the ML Model Intelligence page ─────────────
st.markdown("""
<div class='insight-card'>
    <div class='title'>📊 How the forecast is made — and how accurate it is</div>
    <div class='body'>What the model leans on, its accuracy on held-out data, and the full
    head-to-head against the old CSR method live on the <b>ML Model Intelligence</b> page
    (Briefing section) — so this page can stay focused on the district's numbers.</div>
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
