"""
Page 1 — Overview.

Plain-language summary of the enrollment forecast for the planning meeting:
what it does, next year's numbers for the whole network and for each school,
and the data-coverage notes that answer the Kindergarten / Grade-9 and
missing-history questions.
"""

import pandas as pd
import streamlit as st

from utils.loader import load_data, build_forecast, latest_year, with_network
from utils.forecast import MODEL_MAE, BACKTEST_HEADLINE

# ── Data & forecast ───────────────────────────────────────────────────────────
df = load_data()
fc = with_network(build_forecast(df), df)
LATEST = latest_year(df)
FYEAR = LATEST + 1

# Schools without an assigned network are kept in the district total but not shown
# as a network of their own.
UNASSIGNED = "Unassigned"
fc_net = fc[fc["NETWORK"] != UNASSIGNED]

district_latest   = int(df[df["SCHOOL_YEAR"] == LATEST]["ENROLLMENT"].sum())
district_forecast = int(fc["FORECAST_ENROLLMENT"].sum())
change_pct        = (district_forecast - district_latest) / district_latest * 100
n_schools         = int(fc["SCHOOL_KEY"].nunique())
n_networks        = int(fc_net["NETWORK"].nunique())

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
        for <b style='color:#FFFFFF;'>every school</b>, which then add up to each
        <b style='color:#FFFFFF;'>network</b> and the district as a whole.
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI strip ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
kpis = [
    (f"{district_forecast:,}", f"{FYEAR} Forecast",  "Students expected across the whole district"),
    (f"{change_pct:+.1f}%",    "Expected Change",    f"Compared with {LATEST}"),
    (f"{n_schools:,}",         "Schools Covered",    f"Rolling up into {n_networks} networks"),
    (f"±{MODEL_MAE:.0f}",      "Typical Accuracy",   "How close each school-grade estimate usually lands"),
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
# HOW WE FORECAST — Then vs Now (plain-English methodology)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>How We Forecast — Then vs Now</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>The shift from the old method to the new one, in plain English</div>",
            unsafe_allow_html=True)

then_col, now_col = st.columns(2, gap="large")
with then_col:
    st.markdown(f"""
    <div style='background:#FDF8F0; border:1px solid #E8D9B8; border-top:4px solid #C8973A;
                border-radius:10px; padding:20px 22px; height:100%;
                box-shadow:0 1px 4px rgba(0,48,87,0.06);'>
        <div style='font-size:0.72rem; font-weight:700; color:#C8973A; letter-spacing:0.1em;
                    text-transform:uppercase; margin-bottom:6px;'>Then</div>
        <div style='font-size:1.05rem; font-weight:800; color:#003057; margin-bottom:10px;'>
            The old way — Cohort Survival Rate
        </div>
        <div style='font-size:0.86rem; color:#4A4030; line-height:1.7;'>
            For years, enrollment was projected with a single rule of thumb: take how many students
            were in a grade this year, and assume a similar share will move up to the next grade next
            year — based on how that move has gone in the recent past. It's easy to follow, but it
            leans on that one pattern alone. It can't take in a school's size, its history, or wider
            trends, so small misjudgements quietly build up grade after grade — and in practice it
            tended to <b>predict more students than actually showed up</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)

with now_col:
    st.markdown(f"""
    <div style='background:#F0FAF4; border:1px solid #BFE6CF; border-top:4px solid #22C55E;
                border-radius:10px; padding:20px 22px; height:100%;
                box-shadow:0 1px 4px rgba(0,48,87,0.06);'>
        <div style='font-size:0.72rem; font-weight:700; color:#1B9E5A; letter-spacing:0.1em;
                    text-transform:uppercase; margin-bottom:6px;'>Now</div>
        <div style='font-size:1.05rem; font-weight:800; color:#003057; margin-bottom:10px;'>
            The new way — District Enrollment ML model
        </div>
        <div style='font-size:0.86rem; color:#26402F; line-height:1.7;'>
            The new model learns from <b>seven years of real enrollment</b> across every school and
            grade. Instead of one rule, it weighs many clues together — last year's count, the school's
            overall size, the grade, the network, and each school's own track record — to spot the
            patterns that actually drive enrollment up or down. It still uses the old "how many students
            move up" idea, but as just one clue among many. The result is a forecast that is
            <b>noticeably more accurate and far less prone to over-counting</b>, with a clear sense of
            how confident we can be.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PREDICTIONS — district total, by network, by school
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"<div class='section-header'>{FYEAR} Predictions</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>School forecasts added up the way leaders plan — by school, "
            "by network, and for the district as a whole</div>", unsafe_allow_html=True)

# Whole-district headline
st.markdown(f"""
<div class='info-box'>
    <b>Whole district:</b> {district_latest:,} students enrolled in {LATEST} →
    <b>{district_forecast:,} expected in {FYEAR}</b>
    ({change_pct:+.1f}%, a change of {district_forecast - district_latest:+,} students).
</div>
""", unsafe_allow_html=True)

# By-network breakdown — school predictions summed up to each network (Unassigned hidden)
network_tbl = (fc_net.groupby("NETWORK", as_index=False)
               .agg(**{"Schools": ("SCHOOL_KEY", "nunique"),
                       f"{LATEST} Actual": ("ACTUAL_LATEST", "sum"),
                       f"{FYEAR} Forecast": ("FORECAST_ENROLLMENT", "sum")}))
network_tbl["Change"] = network_tbl[f"{FYEAR} Forecast"] - network_tbl[f"{LATEST} Actual"]
network_tbl["Change %"] = (network_tbl["Change"] / network_tbl[f"{LATEST} Actual"] * 100).round(1)
network_tbl = network_tbl.sort_values(f"{FYEAR} Forecast", ascending=False).rename(columns={"NETWORK": "Network"})

# By-school breakdown — every school stays; blank the label for unassigned schools
school_tbl = (fc.groupby(["SCHOOL_KEY", "SCHOOL_LABEL", "NETWORK"], as_index=False)
              .agg(**{f"{LATEST} Actual": ("ACTUAL_LATEST", "sum"),
                      f"{FYEAR} Forecast": ("FORECAST_ENROLLMENT", "sum")}))
school_tbl["Change"] = school_tbl[f"{FYEAR} Forecast"] - school_tbl[f"{LATEST} Actual"]
school_tbl["Change %"] = (school_tbl["Change"] /
                          school_tbl[f"{LATEST} Actual"].replace(0, pd.NA) * 100).round(1)
school_tbl["NETWORK"] = school_tbl["NETWORK"].replace(UNASSIGNED, "—")
school_tbl = (school_tbl
              .drop(columns=["SCHOOL_KEY"])
              .rename(columns={"SCHOOL_LABEL": "School", "NETWORK": "Network"})
              .sort_values(f"{FYEAR} Forecast", ascending=False))

# How many schools sit outside the network split (kept in the district total)
hidden = fc[fc["NETWORK"] == UNASSIGNED]
n_hidden_schools  = int(hidden["SCHOOL_KEY"].nunique())
n_hidden_students = int(hidden["FORECAST_ENROLLMENT"].sum())

tab_network, tab_school = st.tabs([f"🏙️  By Network ({len(network_tbl)})",
                                   f"🏫  By School ({len(school_tbl):,})"])

_fmt = {"Schools": "{:,.0f}", f"{LATEST} Actual": "{:,.0f}", f"{FYEAR} Forecast": "{:,.0f}",
        "Change": "{:+,.0f}", "Change %": "{:+.1f}%"}
with tab_network:
    st.dataframe(network_tbl.style.format(_fmt, na_rep="—"), hide_index=True, width="stretch")
    if n_hidden_schools:
        st.caption(f"Network totals exclude {n_hidden_schools} schools "
                   f"(~{n_hidden_students:,} students) that have no assigned network. "
                   f"They are still included in the whole-district total above.")
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
