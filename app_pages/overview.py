"""
Page 1 — Overview.

A clean, top-down briefing: next year's number → why it can be trusted →
the breakdown by network/school → coverage caveats → links to the deep-dive
pages. Methodology detail lives on the "Old Way" and "New Model" pages, not here.
"""

import pandas as pd
import streamlit as st

from utils.loader import (load_data, build_forecast, latest_year, with_network,
                          model_vs_baseline)

# ── Data & forecast ───────────────────────────────────────────────────────────
df = load_data()
fc = with_network(build_forecast(df), df)
LATEST = latest_year(df)
FYEAR = LATEST + 1

UNASSIGNED = "Unassigned"
fc_net = fc[fc["NETWORK"] != UNASSIGNED]

district_latest   = int(df[df["SCHOOL_YEAR"] == LATEST]["ENROLLMENT"].sum())
district_forecast = int(fc["FORECAST_ENROLLMENT"].sum())
change_pct        = (district_forecast - district_latest) / district_latest * 100
change_abs        = district_forecast - district_latest
n_schools         = int(fc["SCHOOL_KEY"].nunique())
n_networks        = int(fc_net["NETWORK"].nunique())

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style='background:linear-gradient(135deg,#003057 0%,#00497a 100%);
            border-radius:12px; padding:36px 44px; margin-bottom:26px;
            border-left:6px solid #C8973A;'>
    <div style='font-size:0.8rem; font-weight:700; color:#C8973A; letter-spacing:0.12em;
                text-transform:uppercase; margin-bottom:10px;'>Enrollment Forecast · {FYEAR}</div>
    <div style='font-size:2.1rem; font-weight:800; color:#FFFFFF; line-height:1.2; margin-bottom:14px;'>
        How many students will walk through the door next fall?
    </div>
    <div style='font-size:0.95rem; color:#B8CFDF; line-height:1.7; max-width:880px;'>
        A machine-learning model trained on <b style='color:#FFFFFF;'>seven years of enrollment</b>
        gives a number for every school — which add up to each network and the district. Here's the
        headline, how far it can be trusted, and where to look closer.
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1 — THE NUMBER
# ══════════════════════════════════════════════════════════════════════════════
kpis = [
    (f"{district_forecast:,}", f"{FYEAR} Forecast",  "Students, district-wide"),
    (f"{change_pct:+.1f}%",    "Expected Change",    f"{change_abs:+,} vs {LATEST}"),
    (f"{n_schools:,}",         "Schools",            "Every open school, by grade"),
    (f"{n_networks}",          "Networks",           "Schools roll up into these"),
]
for col, (val, lbl, sub) in zip(st.columns(4), kpis):
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
# 2 — WHY YOU CAN TRUST IT  (live model-vs-baseline, refreshes on retrain)
# ══════════════════════════════════════════════════════════════════════════════
perf = model_vs_baseline(df)
pm = perf["_meta"]
st.markdown("<div class='section-header' style='border-left-color:#1B9E5A;'>"
            "How much better is it than the old method?</div>", unsafe_allow_html=True)
st.markdown(f"<div class='section-sub'>Tested on last year's held-out data ({pm['year']}, "
            f"{pm['n']:,} school-grades) vs the old Cohort Survival Rate method · lower error is better</div>",
            unsafe_allow_html=True)

for col, key in zip(st.columns(len(perf["order"])), perf["order"]):
    mtr = perf[key]
    b, m = mtr["fmt"].format(mtr["baseline"]), mtr["fmt"].format(mtr["model"])
    with col:
        st.markdown(f"""
        <div title="{mtr['help']}" style='background:#F0FAF4; border:1px solid #BFE6CF;
                    border-top:4px solid #1B9E5A; border-radius:12px; padding:16px 12px;
                    text-align:center; height:100%; box-shadow:0 1px 4px rgba(0,48,87,0.06);'>
            <div style='font-size:1.9rem; font-weight:800; color:#1B9E5A; line-height:1;'>{mtr['improvement']:.0f}%</div>
            <div style='font-size:0.66rem; font-weight:800; color:#1B9E5A; letter-spacing:0.08em; margin-top:2px;'>BETTER</div>
            <div style='font-size:0.8rem; font-weight:700; color:#003057; margin-top:9px;'>{mtr['label']}</div>
            <div style='font-size:0.73rem; color:#64748B; margin-top:3px;'>
                {b} <span style='color:#94A3B8;'>&rarr;</span> <b style='color:#1B9E5A;'>{m}</b></div>
        </div>
        """, unsafe_allow_html=True)

st.caption("Hover any card for what it means. Student-count and budget gains run highest because the "
           "old method occasionally over-counts wildly; the % measures are the conservative read.")

st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 3 — THE BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"<div class='section-header'>The {FYEAR} forecast, broken down</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>The same school-level predictions, summed the way leaders plan</div>",
            unsafe_allow_html=True)

network_tbl = (fc_net.groupby("NETWORK", as_index=False)
               .agg(**{"Schools": ("SCHOOL_KEY", "nunique"),
                       f"{LATEST} Actual": ("ACTUAL_LATEST", "sum"),
                       f"{FYEAR} Forecast": ("FORECAST_ENROLLMENT", "sum")}))
network_tbl["Change"] = network_tbl[f"{FYEAR} Forecast"] - network_tbl[f"{LATEST} Actual"]
network_tbl["Change %"] = (network_tbl["Change"] / network_tbl[f"{LATEST} Actual"] * 100).round(1)
network_tbl = network_tbl.sort_values(f"{FYEAR} Forecast", ascending=False).rename(columns={"NETWORK": "Network"})

school_tbl = (fc.groupby(["SCHOOL_KEY", "SCHOOL_LABEL", "NETWORK"], as_index=False)
              .agg(**{f"{LATEST} Actual": ("ACTUAL_LATEST", "sum"),
                      f"{FYEAR} Forecast": ("FORECAST_ENROLLMENT", "sum")}))
school_tbl["Change"] = school_tbl[f"{FYEAR} Forecast"] - school_tbl[f"{LATEST} Actual"]
school_tbl["Change %"] = (school_tbl["Change"] /
                          school_tbl[f"{LATEST} Actual"].replace(0, pd.NA) * 100).round(1)
school_tbl["NETWORK"] = school_tbl["NETWORK"].replace(UNASSIGNED, "—")
school_tbl = (school_tbl.drop(columns=["SCHOOL_KEY"])
              .rename(columns={"SCHOOL_LABEL": "School", "NETWORK": "Network"})
              .sort_values(f"{FYEAR} Forecast", ascending=False))

hidden = fc[fc["NETWORK"] == UNASSIGNED]
n_hidden_schools  = int(hidden["SCHOOL_KEY"].nunique())
n_hidden_students = int(hidden["FORECAST_ENROLLMENT"].sum())

_fmt = {"Schools": "{:,.0f}", f"{LATEST} Actual": "{:,.0f}", f"{FYEAR} Forecast": "{:,.0f}",
        "Change": "{:+,.0f}", "Change %": "{:+.1f}%"}
tab_network, tab_school = st.tabs([f"🏙️  By Network ({len(network_tbl)})",
                                   f"🏫  By School ({len(school_tbl):,})"])
with tab_network:
    st.dataframe(network_tbl.style.format(_fmt, na_rep="—"), hide_index=True, width="stretch")
    if n_hidden_schools:
        st.caption(f"Excludes {n_hidden_schools} schools (~{n_hidden_students:,} students) with no "
                   f"assigned network — still counted in the district total.")
with tab_school:
    st.dataframe(school_tbl.style.format(_fmt, na_rep="—"), hide_index=True, width="stretch", height=420)
    st.download_button("⬇️  Download school-by-school predictions (CSV)",
                       school_tbl.to_csv(index=False).encode("utf-8"),
                       file_name=f"enrollment_school_predictions_{FYEAR}.csv", mime="text/csv")

st.markdown("<hr class='thin'/>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 4 — GOOD TO KNOW  (caveats tucked into an expander to keep the page calm)
# ══════════════════════════════════════════════════════════════════════════════
# Sort grades K→12 (string-safe — "K" can't be cast to int)
_GRADE_ORDER = {"PK": 0, "K": 1, "1": 2, "2": 3, "3": 4, "4": 5, "5": 6, "6": 7,
                "7": 8, "8": 9, "9": 10, "10": 11, "11": 12, "12": 13}
grades_present = sorted({str(g) for g in fc["GRADE"].unique()},
                        key=lambda g: _GRADE_ORDER.get(g, 99))
entry_added = [g for g in ("K", "9") if g in grades_present]

with st.expander("ⓘ  What the forecast covers — and what to keep in mind"):
    notes = [
        ("Grades covered",
         f"The forecast now spans grades {grades_present[0]}–{grades_present[-1]} "
         f"({len(grades_present)} grade levels across every open school)."),
    ]
    if entry_added:
        notes.append((
            "Kindergarten & Grade 9 are newly included",
            "These are entry grades — students arrive from outside the school, so there's no class below them "
            "to learn from. Their projections lean more on school size and area, and should be treated as "
            "<b>preliminary</b> until the model is retrained to include them."))
    notes.append((
        "Where there's no history, we fill carefully",
        "Entry grades (and any first-year gap) get a typical value from past years, clearly marked as "
        "&ldquo;no history available&rdquo; — never the number we're trying to predict."))
    for title, body in notes:
        st.markdown(f"<div style='margin-bottom:10px;'><b style='color:#003057;'>{title}.</b> "
                    f"<span style='color:#334D66; font-size:0.9rem;'>{body}</span></div>",
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 5 — DIG DEEPER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>Dig deeper</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>The full story lives on these pages</div>", unsafe_allow_html=True)

pages = [
    ("#C8973A", "📜", "The Old Way — CSR", "How the long-standing method works, and where it tends to miss."),
    ("#22C55E", "🧠", "The New Model", "What the model leans on, and the accuracy behind the numbers above."),
    ("#4A90C4", "📊", "District & School", "Interactive views — school reports and the custom simulator."),
]
for col, (color, icon, title, body) in zip(st.columns(3), pages):
    with col:
        st.markdown(f"""
        <div style='background:#FFFFFF; border:1px solid #E2E8F0; border-top:4px solid {color};
                    border-radius:12px; padding:18px; height:100%; box-shadow:0 1px 4px rgba(0,48,87,0.06);'>
            <div style='font-size:0.95rem; font-weight:800; color:#003057; margin-bottom:6px;'>{icon}&nbsp; {title}</div>
            <div style='font-size:0.82rem; color:#4A6580; line-height:1.6;'>{body}</div>
        </div>
        """, unsafe_allow_html=True)
