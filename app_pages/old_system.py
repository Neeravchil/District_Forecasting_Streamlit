"""
Page 2 — Old System Methodology (CSR).

Explains how the legacy Cohort-Survival-Rate method projects enrollment, walks
through the arithmetic on a worked example, and shows — using the model-review
backtest — where it systematically misses.
"""

import pandas as pd
import streamlit as st

from utils.forecast import BACKTEST_HEADLINE, BACKTEST_BY_YEAR

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='background:linear-gradient(135deg,#5a4419 0%,#7a5a1e 100%);
            border-radius:12px; padding:40px 48px; margin-bottom:28px;
            border-left:6px solid #C8973A; width:100%;'>
    <div style='font-size:0.8rem; font-weight:700; color:#F0D9A8; letter-spacing:0.12em;
                text-transform:uppercase; margin-bottom:10px;'>The Old System</div>
    <div style='font-size:2.2rem; font-weight:800; color:#FFFFFF; line-height:1.2; margin-bottom:16px;'>
        Cohort Survival Rate (CSR)
    </div>
    <div style='font-size:0.97rem; color:#EADFC6; line-height:1.7;'>
        The legacy SAS method that has driven enrollment budgeting. It is simple, transparent, and
        intuitive — it follows each grade-cohort forward one year — but it carries a structural bias
        that compounds across grades and years.
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# THE IDEA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>The Core Idea</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>Last year's grade becomes this year's next grade — scaled by how "
            "many students &ldquo;survive&rdquo; the move</div>", unsafe_allow_html=True)

col_l, col_r = st.columns([1, 1], gap="large")
with col_l:
    st.markdown("""
    <div style='background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:20px 22px;
                box-shadow:0 1px 4px rgba(0,48,87,0.06);'>
        <div style='font-weight:700; color:#003057; margin-bottom:10px;'>The formula</div>
        <div style='background:#F5F8FB; border-left:4px solid #C8973A; border-radius:6px;
                    padding:14px 16px; font-size:0.92rem; color:#003057; line-height:1.8;'>
            <b>Next-grade enrollment</b><br/>
            = <b>this-grade enrollment</b> × <b>survival rate</b>
        </div>
        <div style='font-size:0.82rem; color:#4A6580; line-height:1.6; margin-top:12px;'>
            The <b>survival rate</b> is the historical ratio of a grade's size this year to its feeder
            grade's size last year — typically averaged over the last 3 years to smooth noise. A rate
            above 1.0 means the cohort grows as it advances; below 1.0 means it shrinks.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_r:
    st.markdown("""
    <div style='background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:20px 22px;
                box-shadow:0 1px 4px rgba(0,48,87,0.06);'>
        <div style='font-weight:700; color:#003057; margin-bottom:10px;'>Worked example</div>
        <div style='font-size:0.85rem; color:#334D66; line-height:1.9;'>
            A school has <b>100</b> sixth-graders this year.<br/>
            Over the last 3 years, 6th&nbsp;→&nbsp;7th held about <b>0.95</b>.<br/>
            CSR projects next year's 7th grade as<br/>
            <span style='color:#003057; font-weight:700;'>100 × 0.95 = 95 students.</span>
        </div>
        <div style='font-size:0.82rem; color:#4A6580; line-height:1.6; margin-top:12px;'>
            Repeat grade-by-grade and sum to get the school, network, and district totals.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HOW IT RUNS — steps
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>How the CSR Pipeline Runs</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>Four repeating steps, applied per school and grade</div>",
            unsafe_allow_html=True)

steps = [
    ("#4A90C4", "1", "Compute survival rates",
     "For every grade transition (1→2, 2→3, …) divide this year's count by last year's feeder count, then average over the last 3 years."),
    ("#8B5CF6", "2", "Roll cohorts forward",
     "Multiply each grade's current enrollment by its survival rate to land next year's count one grade up."),
    ("#C8973A", "3", "Seed the entry grade",
     "The lowest grade has no feeder inside the school, so it is set from a separate trend or a flat assumption — a known weak point."),
    ("#22C55E", "4", "Sum to budget levels",
     "Add the grade projections up to school, network, and district totals that feed the staffing and funding plan."),
]
cols = st.columns(4)
for col, (color, num, title, desc) in zip(cols, steps):
    with col:
        st.markdown(f"""
        <div style='background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:16px;
                    height:100%; box-shadow:0 1px 4px rgba(0,48,87,0.06);'>
            <div style='min-width:32px; height:32px; background:{color}; border-radius:50%;
                        display:flex; align-items:center; justify-content:center;
                        font-weight:800; color:white; margin-bottom:10px;'>{num}</div>
            <div style='font-weight:700; font-size:0.86rem; color:#003057; margin-bottom:5px;'>{title}</div>
            <div style='font-size:0.78rem; color:#4A6580; line-height:1.5;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# WHERE IT BREAKS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>Where CSR Falls Short</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>Why a transparent method still produces a budget-level bias</div>",
            unsafe_allow_html=True)

weaknesses = [
    ("📈", "One signal only",
     "CSR uses just the feeder grade and a survival ratio. It cannot see school size, region trends, governance, or selective-enrollment effects that all move real enrollment."),
    ("➰", "Errors compound",
     "Each grade is built on the projected grade below it. A small bias early in the chain accumulates as the cohort is rolled forward year after year."),
    ("🌊", "Shocks stick",
     "An anomalous year (e.g., the 2022–24 migrant surge) gets baked into the 3-year survival rate and is then projected forward as if it were the new normal."),
    ("🚪", "Entry grades are guesses",
     "The lowest grade in each school has no feeder, so CSR falls back to a flat or trend assumption — exactly where the largest planning errors tend to appear."),
    ("⬆️", "Systematic over-prediction",
     "Across the backtest CSR consistently projected more students than actually enrolled, over-budgeting staff and funding against a declining district."),
    ("🔍", "No uncertainty",
     "CSR returns a single number with no error band, so leaders cannot tell a confident projection from a shaky one."),
]
wcols = st.columns(3)
for i, (icon, title, body) in enumerate(weaknesses):
    with wcols[i % 3]:
        st.markdown(f"""
        <div style='background:#FFF7ED; border:1px solid #FCD9A8; border-left:4px solid #C8973A;
                    border-radius:0 8px 8px 0; padding:14px 16px; margin-bottom:14px;'>
            <div style='font-weight:700; color:#7a5a1e; font-size:0.86rem; margin-bottom:4px;'>{icon}&nbsp; {title}</div>
            <div style='font-size:0.8rem; color:#5a4419; line-height:1.55;'>{body}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# THE EVIDENCE — CSR side of the backtest
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>CSR in the Backtest</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>4-year out-of-sample walk-forward, SY2023–2026 — the CSR column is "
            "what the legacy method would have produced</div>", unsafe_allow_html=True)

hl = pd.DataFrame(
    [(metric, csr, ml) for metric, (csr, ml) in BACKTEST_HEADLINE.items()],
    columns=["Headline metric (district-wide)", "CSR (legacy)", "ML (new)"],
)
st.dataframe(hl, hide_index=True, width="stretch")

by_year = pd.DataFrame(
    [(y, n, mae_csr) for (y, n, mae_csr, *_rest) in BACKTEST_BY_YEAR],
    columns=["School year", "Predictions", "CSR error per school-grade (MAE)"],
)
st.markdown("<div style='font-size:0.82rem; color:#4A6580; margin:6px 0 4px 2px;'>"
            "CSR's average error per school-grade, year by year:</div>", unsafe_allow_html=True)
st.dataframe(by_year.style.format({"Predictions": "{:,}", "CSR error per school-grade (MAE)": "{:.1f}"}),
             hide_index=True, width="stretch")

st.markdown(f"""
<div class='insight-card'>
    <div class='title'>📌 The headline number</div>
    <div class='body'>Across the backtest CSR over-predicted enrollment by
    <b>{BACKTEST_HEADLINE['Total over/under-prediction (students)'][0]} students
    ({BACKTEST_HEADLINE['as % of actual enrollment'][0]})</b>, with a budget error rate of
    <b>{BACKTEST_HEADLINE['Budget error rate (WAPE)'][0]}</b>. The
    <b>ML Model Intelligence</b> page shows how the Gradient-Boosted-Tree model closes that gap.</div>
</div>
""", unsafe_allow_html=True)
