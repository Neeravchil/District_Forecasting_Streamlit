import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from utils.loader import load_data, build_forecast, latest_year, with_network
from utils.forecast import MODEL_MAE, MODEL_RMSE, MODEL_R2

_LAYOUT = dict(
    paper_bgcolor="white", plot_bgcolor="white",
    font=dict(family="Aptos, Nunito Sans, Segoe UI, Arial", size=12, color="#334D66"),
    margin=dict(l=44, r=44, t=20, b=40),
    showlegend=True, legend=dict(orientation="h", y=1.08),
)

ALL_NETWORKS_LABEL = "All Networks (District-Wide)"

# ── Load ──────────────────────────────────────────────────────────────────────
df = load_data()
fc = with_network(build_forecast(df), df)
LATEST = latest_year(df)
FYEAR = LATEST + 1

sorted_networks = sorted(n for n in df["NETWORK"].dropna().unique() if n != "Unassigned")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<p style='font-size:0.82rem; font-weight:700; color:#E8EDF2; "
                "margin:6px 0 4px 0;'>Chart display</p>", unsafe_allow_html=True)
    chart_mode = st.radio("Show", options=["Total", "By grade"], index=0,
                          key="school_chart_mode")

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='background:linear-gradient(135deg,#003057 0%,#00497a 100%);
            border-radius:12px; padding:32px 40px; margin-bottom:28px;
            border-left:6px solid #C8973A;'>
    <div style='font-size:0.72rem; font-weight:700; letter-spacing:0.14em;
                color:#C8973A; text-transform:uppercase; margin-bottom:10px;'>
        Chicago Public Schools · Enrollment Forecasting
    </div>
    <div style='font-size:1.9rem; font-weight:800; color:#FFFFFF; line-height:1.25;'>
        Forecast at any level — District, Network, or School
    </div>
    <div style='font-size:0.92rem; color:#B8CFDF; margin-top:12px; max-width:720px; line-height:1.6;'>
        Drill from the full district down to a single school. At every level the model produces
        a {FYEAR} enrollment projection built up from individual grade cohorts.
        Keep "All" at any step to see the aggregate for that level.
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — NETWORK
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>Step 1 — Choose a Network</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>Select a network to drill in, or keep "
            "\"All Networks\" for a district-wide projection.</div>", unsafe_allow_html=True)

network_options = [ALL_NETWORKS_LABEL] + sorted_networks
selected_network = st.selectbox("Network", options=network_options, index=0,
                                label_visibility="collapsed", key="network_sel")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — SCHOOL
# ══════════════════════════════════════════════════════════════════════════════
if selected_network != ALL_NETWORKS_LABEL:
    reg_df = df[df["NETWORK"] == selected_network].copy()
    # Only list schools that actually have a forecast (open this year); closed
    # schools have no history/projection and would otherwise error on selection.
    schools_in_network = (fc[fc["NETWORK"] == selected_network][["SCHOOL_KEY", "SCHOOL_LABEL"]]
                          .drop_duplicates().sort_values("SCHOOL_LABEL"))
    # Disambiguate any shared names, then map the chosen label back to its key
    # (real school names don't end in their numeric key, so don't parse the string).
    _dups = schools_in_network["SCHOOL_LABEL"].value_counts()
    schools_in_network = schools_in_network.assign(_disp=[
        f"{lbl} (#{int(k)})" if _dups.get(lbl, 0) > 1 else lbl
        for lbl, k in zip(schools_in_network["SCHOOL_LABEL"], schools_in_network["SCHOOL_KEY"])])
    label_to_key = dict(zip(schools_in_network["_disp"],
                            schools_in_network["SCHOOL_KEY"].astype(int)))
    ALL_SCHOOLS_LABEL = f"All Schools — {selected_network}"
    school_options = [ALL_SCHOOLS_LABEL] + schools_in_network["_disp"].tolist()

    st.markdown("<div class='section-header' style='margin-top:12px;'>"
                "Step 2 — Choose a School</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Select a specific school or keep "
                f"\"All Schools\" for a {selected_network} network projection.</div>",
                unsafe_allow_html=True)
    selected_school = st.selectbox("School", options=school_options, index=0,
                                   label_visibility="collapsed", key="school_sel")
else:
    reg_df = df.copy()
    selected_school = None
    ALL_SCHOOLS_LABEL = None

# ── Determine scope ───────────────────────────────────────────────────────────
if selected_network == ALL_NETWORKS_LABEL:
    scope_df, scope_fc = df.copy(), fc.copy()
    scope_label, view_level = "District-Wide", "district"
elif selected_school is None or selected_school == ALL_SCHOOLS_LABEL:
    scope_df = reg_df.copy()
    scope_fc = fc[fc["NETWORK"] == selected_network].copy()
    scope_label, view_level = f"{selected_network} Network", "network"
else:
    key = int(label_to_key[selected_school])
    scope_df = reg_df[reg_df["SCHOOL_KEY"] == key].copy()
    scope_fc = fc[fc["SCHOOL_KEY"] == key].copy()
    scope_label, view_level = selected_school, "school"

st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# KPI STRIP
# ══════════════════════════════════════════════════════════════════════════════
latest_total   = int(scope_df[scope_df["SCHOOL_YEAR"] == LATEST]["ENROLLMENT"].sum())
forecast_total = int(scope_fc["FORECAST_ENROLLMENT"].sum())
change_pct     = (forecast_total - latest_total) / latest_total * 100 if latest_total else 0.0
n_schools      = int(scope_df[scope_df["SCHOOL_YEAR"] == LATEST]["SCHOOL_KEY"].nunique())
n_grades       = int(scope_fc["GRADE"].nunique())

change_clr = "#22C55E" if change_pct >= 0 else "#EF4444"

if view_level == "school":
    kpis = [
        (f"{latest_total:,}",   f"{LATEST} Enrollment",   "Latest actual total",        "#003057"),
        (f"{forecast_total:,}", f"{FYEAR} Projection",    "Across all grades",          "#C8973A"),
        (f"{change_pct:+.1f}%", "Projected Change",       f"{FYEAR} vs {LATEST}",       change_clr),
        (str(n_grades),         "Grades Served",          "Cohorts projected",          "#4A90C4"),
    ]
elif view_level == "network":
    kpis = [
        (f"{latest_total:,}",   f"{LATEST} Enrollment — Network", selected_network,       "#003057"),
        (f"{forecast_total:,}", f"{FYEAR} Projection — Network",  selected_network,       "#C8973A"),
        (f"{change_pct:+.1f}%", "Projected Change",               f"{FYEAR} vs {LATEST}", change_clr),
        (str(n_schools),        "Schools in Network",             "Open schools",         "#4A90C4"),
    ]
else:
    kpis = [
        (f"{latest_total:,}",   f"{LATEST} Enrollment — District", "All open schools",     "#003057"),
        (f"{forecast_total:,}", f"{FYEAR} Projection — District",  "All open schools",     "#C8973A"),
        (f"{change_pct:+.1f}%", "Projected Change",                f"{FYEAR} vs {LATEST}", change_clr),
        (str(n_schools),        "Open Schools",                    "District-wide",        "#4A90C4"),
    ]

c1, c2, c3, c4 = st.columns(4)
for col, (val, lbl, sub, clr) in zip([c1, c2, c3, c4], kpis):
    with col:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value' style='color:{clr};'>{val}</div>
            <div class='metric-label'>{lbl}</div>
            <div class='metric-sub'>{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ENROLLMENT HISTORY + NEXT-YEAR FORECAST
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"<div class='section-header'>Enrollment History &amp; {FYEAR} Forecast "
            f"— {scope_label}</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>Solid bars = actual school-year totals · "
            "Hatched = model projection · Shaded = ±RMSE band</div>", unsafe_allow_html=True)

hist = (scope_df[scope_df["IS_SCHOOL_OPEN"] == 1]
        .groupby("SCHOOL_YEAR")["ENROLLMENT"].sum().reset_index()
        .sort_values("SCHOOL_YEAR"))
hist_x = hist["SCHOOL_YEAR"].astype(int).astype(str).tolist()
hist_y = hist["ENROLLMENT"].tolist()
all_x = hist_x + [str(FYEAR)]

band = MODEL_RMSE * (len(scope_fc) ** 0.5)

fig = go.Figure()

if chart_mode == "By grade":
    # Stacked actual bars by grade + stacked forecast
    grade_hist = (scope_df[scope_df["IS_SCHOOL_OPEN"] == 1]
                  .groupby(["SCHOOL_YEAR", "GRADE_NUMERIC"])["ENROLLMENT"].sum().reset_index())
    grades = sorted(scope_fc["GRADE_NUMERIC"].unique())
    # GRADE_NUMERIC is an internal ordinal (incl. PK/K); map back to the real grade label
    grade_label = dict(zip(scope_df["GRADE_NUMERIC"], scope_df["GRADE"].astype(str)))
    palette = ["#003057", "#0a4d80", "#1167a5", "#2e86ab", "#4A90C4",
               "#6BA3CF", "#C8973A", "#d8ab57", "#e3c07f", "#EF4444",
               "#f06d6d", "#8B5CF6", "#a988f0"]
    for i, g in enumerate(grades):
        gh = grade_hist[grade_hist["GRADE_NUMERIC"] == g]
        ys = [float(gh[gh["SCHOOL_YEAR"] == y]["ENROLLMENT"].sum()) for y in hist["SCHOOL_YEAR"]]
        gf = scope_fc[scope_fc["GRADE_NUMERIC"] == g]["FORECAST_ENROLLMENT"].sum()
        fig.add_trace(go.Bar(
            name=f"Grade {grade_label.get(g, int(g))}", x=all_x, y=ys + [float(gf)],
            marker_color=palette[i % len(palette)],
        ))
    fig.update_layout(barmode="stack")
else:
    fig.add_trace(go.Bar(
        name="Actual enrollment", x=hist_x, y=hist_y, marker_color="#003057",
    ))
    fig.add_trace(go.Bar(
        name=f"{FYEAR} projection", x=[str(FYEAR)], y=[forecast_total],
        marker_color="rgba(200,151,58,0.55)",
        marker_line=dict(color="#C8973A", width=1.5),
        marker_pattern_shape="/", marker_pattern_fgcolor="#C8973A",
    ))
    if hist_x:  # only connect a trend line when there is prior history
        fig.add_trace(go.Scatter(
            name="Projection trend", x=[hist_x[-1], str(FYEAR)],
            y=[hist_y[-1], forecast_total], mode="lines+markers",
            line=dict(color="#C8973A", width=2.5, dash="dash"),
            marker=dict(size=9, color="#C8973A", symbol="circle-open"),
        ))
    fig.add_trace(go.Scatter(
        x=[str(FYEAR), str(FYEAR)], y=[forecast_total - band, forecast_total + band],
        mode="lines", line=dict(color="#C8973A", width=10), opacity=0.18,
        showlegend=False, name="_band",
    ))

fig.add_annotation(
    x=str(FYEAR), y=forecast_total, text=f"  {forecast_total:,}",
    showarrow=False, font=dict(size=12, color="#8a6a1e", weight=700),
    xanchor="left", yanchor="bottom",
)
fig.update_layout(
    **_LAYOUT,
    xaxis=dict(title="School year", categoryorder="array", categoryarray=all_x, showgrid=False),
    yaxis=dict(title="Enrollment", showgrid=True, gridcolor="#F1F5F9", tickformat=","),
    height=460,
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(f"""
<div class='info-box'>
    <b>Model accuracy (trained 2020–{LATEST - 1}, validated on held-out SY{LATEST}):</b>&nbsp;
    MAE = <b>{MODEL_MAE:.1f} students</b> per school-grade
    &nbsp;·&nbsp; RMSE = <b>{MODEL_RMSE:.1f} students</b>
    &nbsp;·&nbsp; R² = <b>{MODEL_R2:.2f}</b>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PER-GRADE FORECAST DETAIL
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"<div class='section-header'>Grade-Level Projection — {FYEAR}</div>",
            unsafe_allow_html=True)
st.markdown(f"<div class='section-sub'>Projected enrollment by grade, summed across the "
            f"current scope · compared with the {LATEST} actual</div>", unsafe_allow_html=True)

grade_tbl = (scope_fc.groupby(["GRADE_NUMERIC", "GRADE"], as_index=False)
             .agg(ACTUAL=("ACTUAL_LATEST", "sum"),
                  FORECAST=("FORECAST_ENROLLMENT", "sum"))
             .sort_values("GRADE_NUMERIC"))
grade_tbl["Change"] = grade_tbl["FORECAST"] - grade_tbl["ACTUAL"]
grade_tbl["Change %"] = (grade_tbl["Change"] / grade_tbl["ACTUAL"].replace(0, pd.NA) * 100)

disp = pd.DataFrame({
    "Grade": grade_tbl["GRADE"].astype(str),
    f"{LATEST} Actual": grade_tbl["ACTUAL"].map(lambda v: f"{v:,.0f}"),
    f"{FYEAR} Forecast": grade_tbl["FORECAST"].map(lambda v: f"{v:,.0f}"),
    "Change": grade_tbl["Change"].map(lambda v: f"{v:+,.0f}"),
    "Change %": grade_tbl["Change %"].map(lambda v: "—" if pd.isna(v) else f"{v:+.1f}%"),
})
# Total row
tot_actual, tot_fc = grade_tbl["ACTUAL"].sum(), grade_tbl["FORECAST"].sum()
tot_chg = tot_fc - tot_actual
disp.loc[len(disp)] = ["TOTAL", f"{tot_actual:,.0f}", f"{tot_fc:,.0f}",
                       f"{tot_chg:+,.0f}",
                       f"{(tot_chg/tot_actual*100):+.1f}%" if tot_actual else "—"]
st.dataframe(disp, hide_index=True, use_container_width=True)

st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PLANNING CALLOUT — sections to plan
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>Sections to Plan</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>Estimated homerooms at common class sizes, applied to the "
            f"{FYEAR} projection (whole rooms, rounded up)</div>", unsafe_allow_html=True)

import math
kc1, kc2, kc3 = st.columns(3)
for col, size in zip([kc1, kc2, kc3], [20, 25, 30]):
    sections = math.ceil(forecast_total / size) if forecast_total else 0
    with col:
        st.markdown(f"""
        <div class='metric-card' style='border-top:3px solid #4A90C4; text-align:left;'>
            <div style='font-size:0.70rem; font-weight:700; letter-spacing:0.12em;
                        color:#4A90C4; text-transform:uppercase; margin-bottom:8px;'>
                {size} students / section
            </div>
            <div style='font-size:1.75rem; font-weight:800; color:#003057;
                        line-height:1.1; margin-bottom:4px;'>{sections:,}</div>
            <div class='metric-label' style='text-transform:none; letter-spacing:0;
                        font-size:0.75rem;'>sections across all grades</div>
        </div>""", unsafe_allow_html=True)

st.markdown(f"""
<div class='info-box' style='margin-top:14px;'>
    <b>How to read this:</b>
    The {FYEAR} projection is built grade-by-grade from each school's cohort history and rolled up to the
    selected scope. Section counts are a planning aid — divide the grade-level projection by your target
    class size and add a small buffer for late registrations before locking staffing.
</div>
""", unsafe_allow_html=True)
