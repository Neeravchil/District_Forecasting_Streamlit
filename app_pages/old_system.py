"""
Page 2 — The Old Way (Cohort Survival Rate).

Explains, in plain language, how the long-standing CSR method projects
enrollment, walks through it on a simple example, and shows where it tends to
miss — using the same four-year test as the new model.
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
                text-transform:uppercase; margin-bottom:10px;'>The Old Way</div>
    <div style='font-size:2.2rem; font-weight:800; color:#FFFFFF; line-height:1.2; margin-bottom:16px;'>
        Cohort Survival Rate (CSR)
    </div>
    <div style='font-size:0.97rem; color:#EADFC6; line-height:1.7;'>
        The method we've used for years to plan enrollment. It's simple and easy to follow — it just
        moves each grade up one year — but it leans on a single signal, and small errors quietly add up
        across grades and years.
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# THE IDEA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>The Basic Idea</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>This year's grade becomes next year's grade up — adjusted for how "
            "many students usually stay</div>", unsafe_allow_html=True)

col_l, col_r = st.columns([1, 1], gap="large")
with col_l:
    st.markdown("""
    <div style='background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:20px 22px;
                height:100%; box-shadow:0 1px 4px rgba(0,48,87,0.06);'>
        <div style='font-weight:700; color:#003057; margin-bottom:10px;'>The rule</div>
        <div style='background:#F5F8FB; border-left:4px solid #C8973A; border-radius:6px;
                    padding:14px 16px; font-size:0.92rem; color:#003057; line-height:1.8;'>
            <b>Next year's grade</b><br/>
            = <b>this year's grade below it</b> × <b>the "stay" rate</b>
        </div>
        <div style='font-size:0.82rem; color:#4A6580; line-height:1.6; margin-top:12px;'>
            The <b>"stay" rate</b> is simply how many students a grade usually keeps as it moves up —
            measured from the last few years. Above 1.0 means a grade tends to grow as it advances;
            below 1.0 means it tends to shrink.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_r:
    st.markdown("""
    <div style='background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:20px 22px;
                height:100%; box-shadow:0 1px 4px rgba(0,48,87,0.06);'>
        <div style='font-weight:700; color:#003057; margin-bottom:10px;'>A simple example</div>
        <div style='font-size:0.85rem; color:#334D66; line-height:1.9;'>
            A school has <b>100</b> sixth-graders this year.<br/>
            Over the last few years, about <b>95%</b> of sixth-graders came back as seventh-graders.<br/>
            So CSR expects next year's seventh grade to be<br/>
            <span style='color:#003057; font-weight:700;'>100 × 0.95 = 95 students.</span>
        </div>
        <div style='font-size:0.82rem; color:#4A6580; line-height:1.6; margin-top:12px;'>
            Do this for every grade and add them up to get the school, network, and district totals.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HOW IT RUNS — steps
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>How It Works, Step by Step</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>The same four steps, repeated for every school and grade</div>",
            unsafe_allow_html=True)

steps = [
    ("#4A90C4", "1", "Measure the “stay” rate",
     "For each move up a grade, look at how many students typically carried over, averaged across the last few years."),
    ("#8B5CF6", "2", "Move each grade up",
     "Multiply this year's grade by its stay rate to get next year's grade above it."),
    ("#C8973A", "3", "Guess the entry grade",
     "The lowest grade has no grade below it to build on, so it's set from a separate trend or a flat assumption — a known weak spot."),
    ("#22C55E", "4", "Add it all up",
     "Total the grades into school, network, and district numbers that feed the staffing and budget plan."),
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
# WHERE IT FALLS SHORT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>Where It Falls Short</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>Why a simple, trusted method still ends up off at budget time</div>",
            unsafe_allow_html=True)

weaknesses = [
    ("📈", "It looks at one thing only",
     "CSR uses just the grade below and a stay rate. It can't see school size, neighborhood trends, or school type — all of which move real enrollment."),
    ("➰", "Small errors snowball",
     "Each grade is built on the projected grade below it, so an early mistake grows as the group is carried forward year after year."),
    ("🌊", "Unusual years stick around",
     "A one-off year — like the 2022–24 newcomer surge — gets baked into the stay rate and then projected forward as if it were normal."),
    ("🚪", "Entry grades are guesses",
     "The lowest grade in each school has nothing below it to build on, so it's essentially estimated — and that's where the biggest misses tend to be."),
    ("⬆️", "It usually predicts too many",
     "Across testing, CSR consistently expected more students than actually showed up — over-budgeting staff and funding against a shrinking district."),
    ("🔍", "No sense of confidence",
     "CSR gives one number with no range, so leaders can't tell a solid projection from a shaky one."),
]
wcols = st.columns(3)
for i, (icon, title, body) in enumerate(weaknesses):
    with wcols[i % 3]:
        st.markdown(f"""
        <div style='background:#FFF7ED; border:1px solid #FCD9A8; border-left:4px solid #C8973A;
                    border-radius:0 8px 8px 0; padding:14px 16px; margin-bottom:14px; min-height:128px;'>
            <div style='font-weight:700; color:#7a5a1e; font-size:0.86rem; margin-bottom:4px;'>{icon}&nbsp; {title}</div>
            <div style='font-size:0.8rem; color:#5a4419; line-height:1.55;'>{body}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# THE EVIDENCE — CSR side of the test
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>How CSR Did in Testing</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>We replayed the last four years (2023–2026) and checked CSR against "
            "what actually happened</div>", unsafe_allow_html=True)

hl = pd.DataFrame(
    [(metric, csr, ml) for metric, (csr, ml) in BACKTEST_HEADLINE.items()],
    columns=["What we measured", "Old way (CSR)", "New model"],
)
st.dataframe(hl, hide_index=True, width="stretch")

by_year = pd.DataFrame(
    [(y, n, mae_csr) for (y, n, mae_csr, *_rest) in BACKTEST_BY_YEAR],
    columns=["School year", "Estimates checked", "CSR — typical miss per school-grade (students)"],
)
st.markdown("<div style='font-size:0.82rem; color:#4A6580; margin:6px 0 4px 2px;'>"
            "How far off CSR was, on average, year by year:</div>", unsafe_allow_html=True)
st.dataframe(
    by_year.style.format({"Estimates checked": "{:,}",
                          "CSR — typical miss per school-grade (students)": "{:.1f}"}),
    hide_index=True, width="stretch")

st.markdown(f"""
<div class='insight-card'>
    <div class='title'>📌 The headline</div>
    <div class='body'>Across the four-year test, CSR expected
    <b>{BACKTEST_HEADLINE['Total over/under-prediction (students)'][0]} more students than actually
    enrolled ({BACKTEST_HEADLINE['as % of actual enrollment'][0]} too high)</b>. The
    <b>New Model</b> page shows how the new approach closes most of that gap.</div>
</div>
""", unsafe_allow_html=True)
