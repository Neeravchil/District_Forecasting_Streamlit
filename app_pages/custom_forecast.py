import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from utils.loader import load_data, latest_year, xgb_predict_row, ENTRY_GRADES
from utils.forecast import predict_rows, FEATURES, GRADE_IDX, GRADE_IDX_KEEP, MODEL_RMSE

_LAYOUT = dict(
    paper_bgcolor="white", plot_bgcolor="white",
    font=dict(family="Aptos, Nunito Sans, Segoe UI, Arial", size=12, color="#334D66"),
    margin=dict(l=44, r=44, t=20, b=40),
    showlegend=True, legend=dict(orientation="h", y=1.08),
)

df = load_data()
LATEST = latest_year(df)
FYEAR = LATEST + 1


@st.cache_data
def _context():
    """District reference values used to pre-fill and to hold hidden features."""
    latest = df[df["SCHOOL_YEAR"] == LATEST]
    grade_lookup = (df[["GRADE", "GRADE_NUMERIC"]].drop_duplicates()
                    .sort_values("GRADE_NUMERIC"))
    grades = grade_lookup["GRADE"].astype(str).tolist()
    grade_to_num = dict(zip(grade_lookup["GRADE"].astype(str),
                            grade_lookup["GRADE_NUMERIC"].astype(int)))
    dist_grade = latest.groupby("GRADE_NUMERIC")["ENROLLMENT"].sum().to_dict()

    gov_enc = (df.dropna(subset=["GOVERNANCE"]).groupby("GOVERNANCE")["GOVERNANCE_ENCODED"]
               .agg(lambda s: int(s.mode().iloc[0])).to_dict())
    # Each network maps to the local-area code the model expects (REGION_ENCODED).
    net_enc = (df.dropna(subset=["NETWORK"]).groupby("NETWORK")["REGION_ENCODED"]
               .agg(lambda s: int(s.mode().iloc[0])).to_dict())

    return {
        "grades": grades,
        "grade_to_num": grade_to_num,
        "dist_grade": dist_grade,
        "gov_enc": gov_enc,
        "net_enc": net_enc,
        "networks": sorted(n for n in net_enc if n != "Unassigned"),
        "governances": [g for g in ["District", "Charter", "Contract", "ALOP", "SAFE"]
                        if g in gov_enc],
        "global_sr": df.attrs.get("global_sr", 1.0),
        "med_avg_sr3": float(df["AVG_SURVIVAL_RATE_3YR"].median()),
        "med_same_grade": int(df.loc[df["SAME_GRADE_LAST_YEAR"].notna(),
                                     "SAME_GRADE_LAST_YEAR"].median()),
        "med_school_total": int(df.loc[df["SCHOOL_TOTAL_LAST_YEAR"].notna(),
                                       "SCHOOL_TOTAL_LAST_YEAR"].median()),
        "med_rep": {c: float(df[c].median()) for c in [
            "EFFECTIVE_LEADERS_LAST_YEAR", "COLLABORATIVE_TEACHERS_LAST_YEAR",
            "INVOLVED_FAMILIES_LAST_YEAR", "SUPPORTIVE_ENVIRONMENT_LAST_YEAR",
            "AMBITIOUS_INSTRUCTION_LAST_YEAR"]},
    }


CTX = _context()

# ── Sidebar: reset ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<p style='font-size:0.82rem; font-weight:700; color:#E8EDF2; "
                "margin:6px 0 4px 0;'>Simulator</p>", unsafe_allow_html=True)
    if st.button("↺  Reset to district defaults", use_container_width=True, key="cf_reset"):
        for k in list(st.session_state.keys()):
            if k.startswith("cf_"):
                del st.session_state[k]
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='background:linear-gradient(135deg,#003057 0%,#00497a 100%);
            border-radius:12px; padding:32px 40px; margin-bottom:28px;
            border-left:6px solid #2E6CA4;'>
    <div style='font-size:0.72rem; font-weight:700; letter-spacing:0.14em;
                color:#2E6CA4; text-transform:uppercase; margin-bottom:10px;'>
        Chicago Public Schools · Custom School Simulator
    </div>
    <div style='font-size:1.9rem; font-weight:800; color:#FFFFFF; line-height:1.25;'>
        Project Any Grade Cohort — Real or Hypothetical
    </div>
    <div style='font-size:0.92rem; color:#B8CFDF; margin-top:12px; max-width:740px; line-height:1.6;'>
        Describe one grade at one school — its recent counts and a few characteristics — and the model
        projects next year's enrollment for that cohort. Inputs are pre-filled with CPS reference values
        so you can start immediately; everything not shown is held at district medians.
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# INPUTS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>Configure the Cohort</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>Adjust the inputs below — all other variables are held "
            "at CPS district medians.</div>", unsafe_allow_html=True)

r1c1, r1c2, r1c3 = st.columns([1.4, 1, 1])
with r1c1:
    school_name = st.text_input("School name (for labelling)",
                                value="Hypothetical School", key="cf_name")
with r1c2:
    grade = st.selectbox("Grade", options=CTX["grades"],
                         index=min(5, len(CTX["grades"]) - 1), key="cf_grade")
with r1c3:
    network = st.selectbox("Network", options=CTX["networks"], key="cf_network")

st.markdown("<br>", unsafe_allow_html=True)

col_l, col_r = st.columns(2)

with col_l:
    st.markdown(f"""
    <div style='background:#F0F5FA; border:1px solid #D1DBE8; border-top:4px solid #003057;
                border-radius:10px; padding:20px 20px 8px 20px; margin-bottom:4px;'>
        <div style='font-size:0.95rem; font-weight:800; color:#003057; margin-bottom:2px;'>
            🎓&nbsp; Cohort History
        </div>
        <div style='font-size:0.76rem; color:#4A6580; margin-bottom:16px;'>
            The single strongest driver of next year's count
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"**This grade — last year ({LATEST})**",
                unsafe_allow_html=True)
    same_last = st.number_input("same_last", min_value=0, max_value=4000,
                                value=CTX["med_same_grade"], step=1, key="cf_same_last",
                                label_visibility="collapsed",
                                help="Students in this grade at this school last year — "
                                     "the model's anchor.")

    st.markdown(f"**This grade — two years ago ({LATEST - 1})**", unsafe_allow_html=True)
    same_2yr = st.number_input("same_2yr", min_value=0, max_value=4000,
                               value=int(round(CTX["med_same_grade"])), step=1,
                               key="cf_same_2yr", label_visibility="collapsed",
                               help="Same grade, two years ago — captures the recent trend.")

    st.markdown(f"**Feeder grade — last year ({LATEST})**",
                unsafe_allow_html=True)
    feeder_last = st.number_input("feeder_last", min_value=0, max_value=4000,
                                  value=CTX["med_same_grade"], step=1, key="cf_feeder_last",
                                  label_visibility="collapsed",
                                  help="Students in the grade just below, last year — they "
                                       "promote into this grade. Set 0 if there is no feeder.")

with col_r:
    st.markdown(f"""
    <div style='background:#EEF4FA; border:1px solid #CFE0F0; border-top:4px solid #2E6CA4;
                border-radius:10px; padding:20px 20px 8px 20px; margin-bottom:4px;'>
        <div style='font-size:0.95rem; font-weight:800; color:#2E6CA4; margin-bottom:2px;'>
            🏫&nbsp; School Context
        </div>
        <div style='font-size:0.76rem; color:#335E85; margin-bottom:16px;'>
            Size and type signals that shape the projection
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"**School total — last year ({LATEST})**",
                unsafe_allow_html=True)
    school_total = st.number_input("school_total", min_value=0, max_value=6000,
                                   value=CTX["med_school_total"], step=10,
                                   key="cf_school_total", label_visibility="collapsed",
                                   help="Total enrollment across all grades at this school last year.")

    st.markdown("**Governance**", unsafe_allow_html=True)
    governance = st.selectbox("governance", options=CTX["governances"], index=0,
                              key="cf_gov", label_visibility="collapsed",
                              help="School operating type.")

    st.markdown("**School level**", unsafe_allow_html=True)
    is_hs = st.radio("school_level", options=["Elementary / Middle", "High School"],
                     index=1 if str(grade) in {"10", "11", "12"} else 0,
                     horizontal=True, key="cf_hs", label_visibility="collapsed")

st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# BUILD FEATURE ROW + PREDICT
# ══════════════════════════════════════════════════════════════════════════════
gn = CTX["grade_to_num"].get(str(grade), 0)
feeder_2yr = same_2yr if feeder_last > 0 else 0      # rough prior for the feeder cohort
has_feeder = 1.0 if feeder_last > 0 else 0.0
csr = (same_last / feeder_2yr) if feeder_2yr > 0 else CTX["global_sr"]
dist_grade_total = CTX["dist_grade"].get(gn, np.median(list(CTX["dist_grade"].values())))

row = {
    "SAME_GRADE_LAST_YEAR": same_last,
    "SAME_GRADE_2YR_AGO": same_2yr,
    "FEEDER_GRADE_LAST_YEAR": feeder_last,
    "FEEDER_GRADE_2YR_AGO": feeder_2yr,
    "SCHOOL_TOTAL_LAST_YEAR": school_total,
    "COHORT_SURVIVAL_RATE": csr,
    "AVG_SURVIVAL_RATE_3YR": CTX["med_avg_sr3"],
    "DISTRICT_GRADE_ENROLLMENT_LAST_YEAR": dist_grade_total,
    # 5Essentials pillar scores — held at district medians (not user inputs)
    **CTX["med_rep"],
    # multi-year averages & year-over-year trends derived from the inputs
    "SAME_GRADE_AVG_3YR": (same_last + same_2yr) / 2 if same_2yr else float(same_last),
    "SAME_GRADE_TREND": float(same_last - same_2yr),
    "FEEDER_GRADE_TREND": float(feeder_last - feeder_2yr),
    "SCHOOL_TOTAL_2YR_AGO": float(school_total),
    "SCHOOL_TOTAL_AVG_3YR": float(school_total),
    "SCHOOL_TOTAL_TREND": 0.0,
    "DISTRICT_GRADE_ENROLLMENT_2YR_AGO": float(dist_grade_total),
    "DISTRICT_GRADE_ENROLLMENT_AVG_3YR": float(dist_grade_total),
    "DISTRICT_GRADE_ENROLLMENT_TREND": 0.0,
    "IS_MIGRANT_ANOMALY_YEAR": 0.0,
    "GRADE_idx": GRADE_IDX.get(str(grade), GRADE_IDX_KEEP),
}
# K & 9 are scored by the dedicated XGBoost model; all other grades by the Spark GBT.
if str(grade) in ENTRY_GRADES:
    forecast = xgb_predict_row(row)
else:
    forecast = float(predict_rows(pd.DataFrame([row]))[0])
change = forecast - same_last
change_pct = (change / same_last * 100) if same_last else 0.0
change_clr = "#22C55E" if change >= 0 else "#EF4444"

# ── Summary KPIs ──────────────────────────────────────────────────────────────
s1, s2, s3, s4 = st.columns(4)
for col, (val, lbl, sub, clr) in zip([s1, s2, s3, s4], [
    (f"Grade {grade}",          "Cohort",               school_name,                 "#003057"),
    (f"{same_last:,}",          f"{LATEST} Actual",     "Entered this year",         "#4A90C4"),
    (f"{forecast:,.0f}",        f"{FYEAR} Projection",  "Model output",              "#2E6CA4"),
    (f"{change_pct:+.1f}%",     "Projected Change",     f"{change:+,.0f} students",  change_clr),
]):
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
# CHART
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"<div class='section-header'>Cohort Trajectory — {school_name}, Grade {grade}</div>",
            unsafe_allow_html=True)
st.markdown("<div class='section-sub'>Your assumed history and the model's "
            f"{FYEAR} projection · shaded band = typical error range</div>", unsafe_allow_html=True)

x = [str(LATEST - 1), str(LATEST), str(FYEAR)]
y_hist = [same_2yr, same_last]
band = MODEL_RMSE

fig = go.Figure()
fig.add_trace(go.Bar(name="Assumed actual", x=x[:2], y=y_hist, marker_color="#003057"))
fig.add_trace(go.Bar(
    name=f"{FYEAR} projection", x=[str(FYEAR)], y=[forecast],
    marker_color="rgba(46,108,164,0.55)", marker_line=dict(color="#2E6CA4", width=1.5),
    marker_pattern_shape="/", marker_pattern_fgcolor="#2E6CA4",
))
fig.add_trace(go.Scatter(
    name="Trend", x=[str(LATEST), str(FYEAR)], y=[same_last, forecast],
    mode="lines+markers", line=dict(color="#2E6CA4", width=2.5, dash="dash"),
    marker=dict(size=9, color="#2E6CA4", symbol="circle-open"),
))
fig.add_trace(go.Scatter(
    x=[str(FYEAR), str(FYEAR)], y=[forecast - band, forecast + band],
    mode="lines", line=dict(color="#2E6CA4", width=12), opacity=0.18,
    showlegend=False, name="_band",
))
fig.add_annotation(x=str(FYEAR), y=forecast, text=f"  {forecast:,.0f}", showarrow=False,
                   font=dict(size=12, color="#23527C", weight=700),
                   xanchor="left", yanchor="bottom")
fig.update_layout(
    **_LAYOUT,
    xaxis=dict(title="School year", categoryorder="array", categoryarray=x, showgrid=False),
    yaxis=dict(title="Grade enrollment", showgrid=True, gridcolor="#F1F5F9", tickformat=","),
    height=440, bargap=0.35,
)
st.plotly_chart(fig, use_container_width=True)

st.markdown(f"""
<div class='info-box' style='margin-top:10px;'>
    <b>How this works.</b>
    Your inputs describe one grade's recent size and a few school characteristics; the same model used
    across this dashboard then projects next year's count. Anything you don't set is held at a typical
    CPS value. The shaded band shows the typical error (about {MODEL_RMSE:.0f} students), so treat the
    projection as a planning estimate, not an exact figure.
</div>
""", unsafe_allow_html=True)
