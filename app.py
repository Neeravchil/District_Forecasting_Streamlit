import streamlit as st

st.set_page_config(
    page_title="Enrollment Forecasting | CPS",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito+Sans:wght@300;400;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0');

html, body, [class*="css"], * {
    font-family: Aptos, 'Nunito Sans', 'Segoe UI', Arial, sans-serif !important;
}

.block-container { padding-top: 1rem !important; }
#MainMenu, footer { visibility: hidden; }

/* ── Fix: restore Material Symbols Rounded font on icon spans ── */
[data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded' !important;
    font-feature-settings: 'liga' !important;
    -webkit-font-feature-settings: 'liga' !important;
    -webkit-font-smoothing: antialiased !important;
}

/* ── Sidebar collapse / expand toggle buttons ── */
button[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"] button {
    background-color: rgba(46, 108, 164, 0.15) !important;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%232E6CA4'%3E%3Cpath d='M21 3H3c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h18c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H3V5h18v14zM9 7H5v2h4V7zm0 4H5v2h4v-2zm0 4H5v2h4v-2zm10-8h-8v2h8V7zm0 4h-8v2h8v-2z'/%3E%3C/svg%3E") !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    background-size: 20px 20px !important;
    color: transparent !important;
    border: 1px solid rgba(46, 108, 164, 0.50) !important;
    border-radius: 6px !important;
    width: 36px !important;
    height: 36px !important;
    min-width: 36px !important;
    min-height: 36px !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    opacity: 1 !important;
    visibility: visible !important;
    overflow: hidden !important;
    cursor: pointer !important;
    transition: background-color 0.2s !important;
}
button[data-testid="stExpandSidebarButton"] > *,
[data-testid="stSidebarCollapseButton"] button > * {
    display: none !important;
}
button[data-testid="stExpandSidebarButton"]:hover,
[data-testid="stSidebarCollapseButton"] button:hover {
    background-color: rgba(46, 108, 164, 0.35) !important;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%23FFFFFF'%3E%3Cpath d='M21 3H3c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h18c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H3V5h18v14zM9 7H5v2h4V7zm0 4H5v2h4v-2zm0 4H5v2h4v-2zm10-8h-8v2h8V7zm0 4h-8v2h8v-2z'/%3E%3C/svg%3E") !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    background-size: 20px 20px !important;
}

/* ════════════════════════════════════════════════════
   SIDEBAR — navy background
   ════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #003057 0%, #00213d 100%) !important;
    border-right: 3px solid #2E6CA4 !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] a,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #E8EDF2 !important;
}
[data-testid="stSidebarNavItems"] a {
    border-radius: 6px !important;
    padding: 8px 12px !important;
    transition: background 0.2s;
}
[data-testid="stSidebarNavItems"] a:hover {
    background: rgba(46, 108, 164, 0.18) !important;
}
[data-testid="stSidebarNavItems"] [aria-current="page"],
[data-testid="stSidebarNavItems"] [aria-selected="true"] {
    background: rgba(46, 108, 164, 0.25) !important;
    border-left: 3px solid #2E6CA4 !important;
}

/* ── Selectbox / multiselect container ── */
[data-testid="stSidebar"] [data-baseweb="select"] > div:first-child {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(46, 108, 164, 0.40) !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] {
    background: rgba(46, 108, 164, 0.30) !important;
    border: 1px solid rgba(46, 108, 164, 0.60) !important;
    border-radius: 4px !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] span { color: #FFFFFF !important; }
[data-testid="stSidebar"] [data-baseweb="tag"] svg { fill: #FFFFFF !important; color: #FFFFFF !important; }
[data-testid="stSidebar"] [data-baseweb="select"] input,
[data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdown"] {
    color: #E8EDF2 !important;
    caret-color: #2E6CA4 !important;
}

/* ── Slider ── */
[data-testid="stSidebar"] [data-testid="stSlider"] div[class*="StyledSliderTrack"] {
    background: rgba(46, 108, 164, 0.30) !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] div[class*="StyledSliderTrack"] > div {
    background: #2E6CA4 !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
    background: #2E6CA4 !important;
    border: 2px solid #FFFFFF !important;
    box-shadow: 0 0 0 3px rgba(46,108,164,0.30) !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: #B8CFDF !important;
}

/* ── Sidebar button ── */
[data-testid="stSidebar"] .stButton > button {
    background: rgba(46, 108, 164, 0.15) !important;
    border: 1px solid rgba(46, 108, 164, 0.50) !important;
    color: #DCEAF6 !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    width: 100% !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(46, 108, 164, 0.35) !important;
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(46, 108, 164, 0.35) !important; }
[data-testid="stSidebar"] [data-testid="stImage"] { padding: 0 !important; margin: 0 !important; }
[data-testid="stSidebar"] [data-testid="stImage"] img { display: block !important; margin: 0 !important; }

/* ════════════════════════════════════════════════════
   MAIN CONTENT — shared component styles
   ════════════════════════════════════════════════════ */
.metric-card {
    background: #FFFFFF;
    border: 1px solid #D1DBE8;
    border-top: 3px solid #003057;
    border-radius: 8px;
    padding: 22px 18px;
    text-align: center;
    box-shadow: 0 2px 6px rgba(0,48,87,0.08);
}
.metric-card .metric-value { font-size: 2rem; font-weight: 800; color: #003057; line-height: 1.1; }
.metric-card .metric-label {
    font-size: 0.76rem; font-weight: 700; color: #003057;
    text-transform: uppercase; letter-spacing: 0.06em; margin-top: 6px;
}
.metric-card .metric-sub { font-size: 0.73rem; color: #64748B; margin-top: 4px; }

.section-header {
    font-size: 1.1rem; font-weight: 700; color: #003057;
    margin-bottom: 4px; border-left: 4px solid #2E6CA4; padding-left: 12px;
}
.section-sub { font-size: 0.82rem; color: #4A6580; margin-bottom: 16px; padding-left: 16px; }

.insight-card {
    background: #F5F8FB; border-left: 4px solid #4A90C4;
    border-radius: 0 8px 8px 0; padding: 14px 18px; margin-bottom: 10px;
}
.insight-card .title { font-weight: 700; color: #003057; font-size: 0.9rem; }
.insight-card .body  { font-size: 0.82rem; color: #334D66; margin-top: 4px; }

.info-box {
    background: #EBF3FA; border: 1px solid #4A90C4; border-radius: 6px;
    padding: 12px 16px; font-size: 0.82rem; color: #003057; margin-bottom: 12px;
}
hr.thin { border: none; border-top: 1px solid #D1DBE8; margin: 20px 0; }

.stButton > button[kind="primary"] {
    background: #003057 !important; border: none !important; color: white !important;
    border-radius: 6px !important; font-weight: 700 !important;
}
.stButton > button[kind="primary"]:hover { background: #2E6CA4 !important; color: #003057 !important; }
.stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar header ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='
        border-left: 4px solid #2E6CA4;
        border-right: 4px solid #2E6CA4;
        margin: 0 0 0 0;
        overflow: hidden;
        line-height: 0;
    '>
    """, unsafe_allow_html=True)
    st.image("static/cps_logo.jpg", use_container_width=True)
    st.markdown("""
    </div>
    <div style='text-align:center; padding:10px 16px 14px 16px;'>
        <div style='font-size:1.0rem; font-weight:800; color:#FFFFFF; line-height:1.35;'>
            Enrollment Forecasting
        </div>
        <div style='font-size:0.72rem; color:#B8CFDF; margin-top:4px;'>
            Chicago Public Schools &nbsp;·&nbsp; 2026–2027 Projection
        </div>
    </div>
    <hr style='border:none; border-top:1px solid rgba(46,108,164,0.4); margin:0 0 10px 0;'/>
    """, unsafe_allow_html=True)

pg = st.navigation(
    {
        "Summary": [
            st.Page("app_pages/overview.py",        title="Overview",                 icon="📋", default=True),
        ],
        "Explore": [
            st.Page("app_pages/district.py",        title="District Overview",        icon="📊"),
            st.Page("app_pages/school_forecast.py", title="School Enrollment Report",  icon="🏫"),
        ],
    },
    position="sidebar",
)

with st.sidebar:
    st.markdown("""
    <hr style='border:none; border-top:1px solid rgba(46,108,164,0.3); margin:6px 0;'/>
    """, unsafe_allow_html=True)
    if st.button("🔄  Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

pg.run()
