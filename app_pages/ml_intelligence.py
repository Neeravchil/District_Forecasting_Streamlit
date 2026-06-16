"""
Page 3 — The New Model.

Plain-language showcase of the new forecasting model: how it's built, what
information it leans on most, how accurate it is, and how it compares with the
old CSR method over the four-year test.
"""

import pandas as pd
import plotly.graph_objects as go

import streamlit as st

from utils.forecast import (OFFICIAL_FI, BACKTEST_HEADLINE, BACKTEST_BY_YEAR,
                            MODEL_MAE, MODEL_RMSE, MODEL_R2)
from utils.loader import accuracy_by_group

CSR_GOLD = "#C8973A"
ML_GREEN  = "#22A36B"
_GRADE_ORDER = {"PK": 0, "K": 1, "1": 2, "2": 3, "3": 4, "4": 5, "5": 6, "6": 7,
                "7": 8, "8": 9, "9": 10, "10": 11, "11": 12, "12": 13}


def _grouped_bar(labels, csr_vals, ml_vals, *, ytitle, fmt="{:.0f}", height=360,
                 rotate=False, log=False):
    """A CSR-gold vs ML-green grouped bar in the page's house style.

    ``log=True`` puts the y-axis on a log scale so one tall bar (e.g. Grade 9,
    whose entry-grade error dwarfs every other grade) can't hide the rest.
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Old way (CSR)", x=labels, y=csr_vals, marker_color=CSR_GOLD,
                         text=[fmt.format(v) for v in csr_vals], textposition="outside",
                         textfont=dict(size=10)))
    fig.add_trace(go.Bar(name="New model", x=labels, y=ml_vals, marker_color=ML_GREEN,
                         text=[fmt.format(v) for v in ml_vals], textposition="outside",
                         textfont=dict(size=10)))
    yaxis = dict(title=ytitle, showgrid=True, gridcolor="#F1F5F9")
    if log:
        yaxis["type"] = "log"
    else:
        top = max(list(csr_vals) + list(ml_vals)) if len(csr_vals) else 1
        yaxis["range"] = [0, top * 1.18]
    fig.update_layout(
        barmode="group", paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Aptos, Nunito Sans, Segoe UI, Arial", size=12, color="#334D66"),
        margin=dict(l=46, r=24, t=24, b=90 if rotate else 40),
        legend=dict(orientation="h", y=1.16, x=0),
        xaxis=dict(showgrid=False, tickangle=-40 if rotate else 0),
        yaxis=yaxis,
        height=height,
    )
    return fig

# Plain-language names for the pieces of information the model uses
FEATURE_NAMES = {
    "SAME_GRADE_LAST_YEAR":               "This grade, last year",
    "DISTRICT_GRADE_ENROLLMENT_LAST_YEAR":"District-wide size of this grade, last year",
    "SCHOOL_TOTAL_LAST_YEAR":             "Whole school's size, last year",
    "GRADE_idx":                          "Which grade it is",
    "GRADE_NUMERIC":                      "Grade order",
    "FEEDER_GRADE_LAST_YEAR":             "Grade below, last year",
    "FEEDER_GRADE_2YR_AGO":               "Grade below, two years ago",
    "COHORT_SURVIVAL_RATE":               "How many students usually stay",
    "AVG_SURVIVAL_RATE_3YR":              "How many usually stay (3-yr average)",
    "SCHOOL_EFFECT":                      "This school's track record",
    "SAME_GRADE_2YR_AGO":                 "This grade, two years ago",
    "REGION_ENCODED":                     "Local area",
    "GOVERNANCE_ENCODED":                 "School type",
    "IS_HIGH_SCHOOL":                     "Is a high school",
    "IS_SELECTIVE":                       "Is selective-enrollment",
    "IS_ATTENDANCE_AREA":                 "Is a neighborhood school",
    "IS_SMALL_SCHOOL":                    "Is a small school",
    "HAS_FEEDER_GRADE":                   "Has a grade below to learn from",
    "IS_MIGRANT_ANOMALY_YEAR":            "Was an unusual (surge) year",
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
        A Model That Learns the Patterns
    </div>
    <div style='font-size:0.97rem; color:#C9EAD9; line-height:1.7;'>
        Instead of relying on a single rule, the new model learns from
        <b style='color:#FFFFFF;'>seven years of real enrollment</b> — picking up how a school's size,
        the grade, the network, and its history all combine to shape next year's numbers. It still uses
        the old "how many students stay" idea, but as <b style='color:#FFFFFF;'>one clue among many</b>.
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HOW IT'S BUILT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>How It's Built</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>From years of records to a number for every school and grade</div>",
            unsafe_allow_html=True)

stages = [
    ("#4A90C4", "1", "Gather 7 years of records",
     "One line for every school and grade, each year, counting how many students were enrolled."),
    ("#8B5CF6", "2", "Use only what we knew at the time",
     "Every clue the model uses comes from earlier years, never the number it's trying to predict — so testing is fair."),
    ("#C8973A", "3", "Learn the patterns",
     "The model studies the records and learns which combinations of clues lead to which outcomes, with unusual surge years turned down so they don't distort normal patterns."),
    ("#EF4444", "4", "Runs anywhere, instantly",
     "The trained model is saved in a lightweight form and produces a forecast in a fraction of a second — no special big-data setup needed."),
    ("#22C55E", "5", "Add up to any level",
     "Grade-by-grade forecasts roll up cleanly to the school, network, or whole-district total."),
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
            <div style='font-weight:700; font-size:0.82rem; color:#003057; margin-bottom:5px;'>{title}</div>
            <div style='font-size:0.76rem; color:#4A6580; line-height:1.45;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# WHAT IT LEANS ON — feature importance
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>What the Model Leans On Most</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>How much each piece of information contributes to the forecast "
            "(totals 100%) — top 10</div>", unsafe_allow_html=True)

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
    <div class='title'>📌 History matters most — but it isn't the whole story</div>
    <div class='body'>The strongest single clue is how big this grade was last year (about a third of the
    forecast). What makes the model better than the old way is the next chunk — the district-wide trend
    for that grade, the school's overall size, and which grade it is — context the old method never
    looked at. The "how many students stay" idea is still in there, just as one clue among many.</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ACCURACY
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>How Accurate Is It?</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>Trained on 2020–2025, then tested on 2026 — a year it had never "
            "seen — for every school and grade</div>", unsafe_allow_html=True)

acc = st.columns(4)
accuracy_kpis = [
    ("#22A36B", f"±{MODEL_MAE:.1f}", "Typical miss",
     "On average, each school-grade estimate lands about 8 students away from the real number."),
    ("#003057", f"{MODEL_RMSE:.0f}", "Worst-case pull",
     "A few very large schools stretch the average error higher — this captures those bigger misses."),
    ("#C8973A", f"{MODEL_R2:.0%}", "Of the ups & downs explained",
     "The model captures most of the real rises and falls in enrollment from school to school."),
    ("#8B5CF6", "100", "Patterns combined",
     "The forecast blends 100 simple decision rules, each correcting the one before it."),
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
# HEAD TO HEAD — new vs old, by year
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>New Model vs the Old Way</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>Typical miss per school-grade across the four-year test "
            "(shorter bars are better)</div>", unsafe_allow_html=True)

years   = [str(r[0]) for r in BACKTEST_BY_YEAR]
mae_csr = [r[2] for r in BACKTEST_BY_YEAR]
mae_ml  = [r[3] for r in BACKTEST_BY_YEAR]

fig = go.Figure()
fig.add_trace(go.Bar(name="Old way (CSR)", x=years, y=mae_csr, marker_color="#C8973A",
                     text=[f"{v:.1f}" for v in mae_csr], textposition="outside"))
fig.add_trace(go.Bar(name="New model", x=years, y=mae_ml, marker_color="#22A36B",
                     text=[f"{v:.1f}" for v in mae_ml], textposition="outside"))
fig.update_layout(
    barmode="group", paper_bgcolor="white", plot_bgcolor="white",
    font=dict(family="Aptos, Nunito Sans, Segoe UI, Arial", size=12, color="#334D66"),
    margin=dict(l=44, r=44, t=20, b=40),
    legend=dict(orientation="h", y=1.12),
    xaxis=dict(title="School year", showgrid=False),
    yaxis=dict(title="Typical miss per school-grade (students)", showgrid=True, gridcolor="#F1F5F9"),
    height=420,
)
st.plotly_chart(fig, width="stretch")

hl = pd.DataFrame(
    [(metric, csr, ml) for metric, (csr, ml) in BACKTEST_HEADLINE.items()],
    columns=["What we measured (four-year test)", "Old way (CSR)", "New model"],
)
st.dataframe(hl, hide_index=True, width="stretch")

st.markdown(f"""
<div class='insight-card'>
    <div class='title'>📌 Roughly a 3× improvement — and it stops over-counting</div>
    <div class='body'>The new model cuts the typical miss per school-grade from about
    <b>{BACKTEST_HEADLINE['Average error per school-grade (MAE)'][0]}</b> students down to about
    <b>{BACKTEST_HEADLINE['Average error per school-grade (MAE)'][1]}</b>, and turns the old method's
    <b>{BACKTEST_HEADLINE['as % of actual enrollment'][0]}</b> over-count into a small
    <b>{BACKTEST_HEADLINE['as % of actual enrollment'][1]}</b> — slightly cautious instead of badly
    over-budgeted.</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# WHERE THE MODEL WINS — by grade (entry grades expose CSR), then by network
# (live on last year's held-out data — refreshes automatically on retrain)
# ══════════════════════════════════════════════════════════════════════════════
acc_grade = accuracy_by_group("GRADE")
acc_net   = accuracy_by_group("NETWORK")
ag = acc_grade.copy()
ag["_o"] = ag["GRADE"].astype(str).map(_GRADE_ORDER).fillna(99)
ag = ag.sort_values("_o")
g_labels = ["Grade " + str(g) for g in ag["GRADE"]]

st.markdown("<div class='section-header'>Where the Model Wins — Grade by Grade</div>",
            unsafe_allow_html=True)
st.markdown("<div class='section-sub'>Last year's held-out test, every grade. The gap is widest at the "
            "<b>entry grades</b> — Kindergarten and Grade 9 — where the old method is flying blind</div>",
            unsafe_allow_html=True)

# Why CSR breaks at entry grades — reactive vs proactive framing
st.markdown("""
<div style='display:flex; gap:14px; margin:6px 0 18px;'>
  <div style='flex:1; background:#FBF6EC; border:1px solid #EAD9B6; border-left:4px solid #C8973A;
              border-radius:10px; padding:14px 16px;'>
    <div style='font-weight:800; color:#8A6A22; font-size:0.86rem; margin-bottom:4px;'>📜 Old way — reactive</div>
    <div style='font-size:0.82rem; color:#5E5024; line-height:1.55;'>Just carries this year's class forward
    by a survival rate. At an <b>entry grade</b> there's no class below to carry — so it falls back on a
    stand-in number and can miss by hundreds of students.</div>
  </div>
  <div style='flex:1; background:#F0FAF4; border:1px solid #BFE6CF; border-left:4px solid #22A36B;
              border-radius:10px; padding:14px 16px;'>
    <div style='font-weight:800; color:#1B7A47; font-size:0.86rem; margin-bottom:4px;'>🧠 New model — proactive</div>
    <div style='font-size:0.82rem; color:#1F5135; line-height:1.55;'>Reads early signals — district-wide
    grade trend, school size, neighbourhood and school type — to anticipate the incoming class
    <b>a year ahead</b>, even with no feeder cohort to lean on.</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Students-off (MAE) on a log scale so Grade 9's spike can't hide the other grades
st.markdown("<div style='font-weight:700; color:#003057; font-size:0.9rem; margin-bottom:2px;'>"
            "Typical miss in students — log scale, so every grade is readable</div>",
            unsafe_allow_html=True)
st.plotly_chart(_grouped_bar(g_labels, ag["csr_mae"], ag["ml_mae"],
                             ytitle="Students off (MAE, log scale)", fmt="{:,.0f}", height=360,
                             rotate=True, log=True),
                width="stretch")

# Same comparison as a percentage — naturally on one scale, the fair read
st.markdown("<div style='font-weight:700; color:#003057; font-size:0.9rem; margin:8px 0 2px;'>"
            "The same gap as a percentage error</div>", unsafe_allow_html=True)
st.plotly_chart(_grouped_bar(g_labels, ag["csr_medape"], ag["ml_medape"],
                             ytitle="Typical % error (MedAPE)", fmt="{:.0f}%", height=360,
                             rotate=True),
                width="stretch")

# Proactive-payoff insight, numbers pulled live from the entry grades
def _mae(grade):
    r = ag[ag["GRADE"].astype(str) == grade]
    return (float(r["csr_mae"].iloc[0]), float(r["ml_mae"].iloc[0])) if len(r) else (0.0, 0.0)
k_csr, k_ml = _mae("K")
n9_csr, n9_ml = _mae("9")
st.markdown(f"""
<div class='insight-card'>
    <div class='title'>📌 The proactive payoff is biggest where it matters most</div>
    <div class='body'>At the continuing grades both methods are close. At the <b>entry grades</b> the old
    method missed Kindergarten by about <b>{k_csr:,.0f}</b> students per school and Grade 9 by about
    <b>{n9_csr:,.0f}</b>; the model pulls those down to roughly <b>{k_ml:,.0f}</b> and <b>{n9_ml:,.0f}</b>.
    That's exactly the incoming class leaders most need to size for — so staffing and budgets can be set
    <b>before</b> the year starts, not corrected after the students arrive.</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

# ── And it holds across the networks (MedAPE — scale-free) ─────────────────────
st.markdown("<div class='section-header'>And It Holds Across the Networks</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>Typical % error by network — the model is more accurate in nearly "
            "every grouping leaders plan around</div>", unsafe_allow_html=True)
an = acc_net[acc_net["NETWORK"].astype(str).str.lower().ne("unassigned")].copy()
an = an.sort_values("csr_medape", ascending=False)
st.plotly_chart(_grouped_bar(list(an["NETWORK"].astype(str)), an["csr_medape"], an["ml_medape"],
                             ytitle="Typical % error (MedAPE)", fmt="{:.0f}%", height=440, rotate=True),
                width="stretch")
st.caption("Live on last year's held-out data — Kindergarten and Grade 9 are scored by the dedicated "
           "entry-grade model, every other grade by the main model. Refreshes automatically on retrain.")
