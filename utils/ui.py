"""
UI Layout and Custom Styling Module for ClimaXplore.

Provides logo header, custom CSS theme styles, navigation indicators, and footer.
"""

import streamlit as st


CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* Main Theme */
    .stApp {
        background-color: #0a0f1e;
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
    }

    /* Top Streamlit Header Bar Override — Darken bar & enhance text visibility */
    [data-testid="stHeader"], header[data-testid="stHeader"], header {
        background-color: #0a0f1e !important;
        background: #0a0f1e !important;
        color: #f1f5f9 !important;
    }
    [data-testid="stHeader"] *, header * {
        color: #f1f5f9 !important;
    }
    [data-testid="stDecoration"] {
        background-image: linear-gradient(90deg, #38bdf8, #2dd4bf) !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #0a0f1e 100%);
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] .css-1d391kg { padding: 1rem 0.75rem; }

    /* Header Styling */
    .cx-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 18px 28px;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        margin-bottom: 28px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
    }

    .cx-logo-box {
        display: flex;
        align-items: center;
        gap: 16px;
    }

    .cx-logo-icon {
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, #38bdf8, #2dd4bf);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        font-weight: 900;
        color: #0a0f1e;
        box-shadow: 0 4px 16px rgba(56,189,248,0.35);
    }

    .cx-logo-title {
        font-size: 26px;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(90deg, #f8fafc, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .cx-logo-subtitle {
        font-size: 13px;
        color: #64748b;
        margin-top: 2px;
    }

    /* Card Container */
    .cx-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    }

    .cx-card:hover {
        border-color: #475569;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
        transition: all 0.2s ease;
    }

    /* Metric Cards Row */
    .cx-metric-row {
        display: flex;
        gap: 16px;
        margin-bottom: 24px;
        flex-wrap: wrap;
    }

    .cx-metric-box {
        flex: 1;
        min-width: 140px;
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }

    .cx-metric-label {
        font-size: 11px;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .cx-metric-val {
        font-size: 26px;
        font-weight: 800;
        color: #38bdf8;
        margin-top: 6px;
        letter-spacing: -0.02em;
    }

    .cx-metric-sub {
        font-size: 11px;
        color: #475569;
        margin-top: 4px;
    }

    /* Phase Badges */
    .cx-badge {
        display: inline-block;
        padding: 5px 14px;
        font-size: 12px;
        font-weight: 600;
        border-radius: 9999px;
        background-color: rgba(56, 189, 248, 0.1);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.25);
        letter-spacing: 0.02em;
    }

    .cx-badge-green {
        background-color: rgba(74, 222, 128, 0.1);
        color: #4ade80;
        border-color: rgba(74, 222, 128, 0.25);
    }

    .cx-badge-amber {
        background-color: rgba(251, 191, 36, 0.1);
        color: #fbbf24;
        border-color: rgba(251, 191, 36, 0.25);
    }

    .cx-badge-red {
        background-color: rgba(248, 113, 113, 0.1);
        color: #f87171;
        border-color: rgba(248, 113, 113, 0.25);
    }

    /* Phase Section Header */
    .cx-phase-header {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 20px;
        background: linear-gradient(90deg, rgba(56,189,248,0.08), transparent);
        border-left: 3px solid #38bdf8;
        border-radius: 0 8px 8px 0;
        margin: 24px 0 16px 0;
    }

    .cx-phase-label {
        font-size: 13px;
        font-weight: 700;
        color: #38bdf8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    .cx-phase-title {
        font-size: 20px;
        font-weight: 700;
        color: #f1f5f9;
    }

    /* Info / Status Boxes */
    .cx-info-box {
        padding: 14px 18px;
        border-radius: 10px;
        margin: 12px 0;
        border: 1px solid;
    }

    .cx-info-box.success {
        background: rgba(74,222,128,0.08);
        border-color: rgba(74,222,128,0.3);
        color: #4ade80;
    }

    .cx-info-box.warning {
        background: rgba(251,191,36,0.08);
        border-color: rgba(251,191,36,0.3);
        color: #fbbf24;
    }

    .cx-info-box.error {
        background: rgba(248,113,113,0.08);
        border-color: rgba(248,113,113,0.3);
        color: #f87171;
    }

    /* Hero Section */
    .cx-hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 20px;
        padding: 48px 40px;
        text-align: center;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    }

    .cx-hero::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(ellipse at center, rgba(56,189,248,0.05) 0%, transparent 70%);
        pointer-events: none;
    }

    .cx-hero-title {
        font-size: 48px;
        font-weight: 900;
        letter-spacing: -0.04em;
        background: linear-gradient(135deg, #f8fafc 30%, #38bdf8 70%, #2dd4bf 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 16px;
        line-height: 1.1;
    }

    .cx-hero-subtitle {
        font-size: 18px;
        color: #64748b;
        margin-bottom: 32px;
        font-weight: 400;
    }

    /* Feature Cards Grid */
    .cx-feature-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin: 24px 0;
    }

    .cx-feature-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        transition: all 0.2s ease;
    }

    .cx-feature-card:hover {
        border-color: #38bdf8;
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(56,189,248,0.1);
    }

    .cx-feature-icon { font-size: 28px; margin-bottom: 10px; }
    .cx-feature-title { font-size: 15px; font-weight: 700; color: #f1f5f9; margin-bottom: 6px; }
    .cx-feature-desc { font-size: 13px; color: #64748b; line-height: 1.5; }

    /* Ranking Table */
    .cx-rank-table { width: 100%; border-collapse: collapse; }
    .cx-rank-table th {
        background: #1e293b;
        color: #94a3b8;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 10px 16px;
        text-align: left;
        border-bottom: 1px solid #334155;
    }
    .cx-rank-table td {
        padding: 10px 16px;
        border-bottom: 1px solid #1e293b;
        font-size: 14px;
        color: #e2e8f0;
    }
    .cx-rank-table tr:hover td { background: rgba(56,189,248,0.04); }
    .cx-rank-first { color: #fbbf24; font-weight: 800; }
    .cx-rank-second { color: #94a3b8; font-weight: 700; }
    .cx-rank-third { color: #b45309; font-weight: 700; }

    /* Footer Styling */
    .cx-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: linear-gradient(90deg, #0f172a, #1e293b, #0f172a);
        color: #475569;
        text-align: center;
        padding: 10px 16px;
        font-size: 12px;
        font-weight: 500;
        border-top: 1px solid #1e293b;
        z-index: 999;
        letter-spacing: 0.02em;
    }

    /* Streamlit overrides */
    .stTabs [data-baseweb="tab-list"] {
        background: #0f172a;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #64748b;
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: #1e293b !important;
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #38bdf8;
        font-size: 1.6rem;
        font-weight: 800;
    }

    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9, #06b6d4);
        color: #0a0f1e;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(14,165,233,0.3);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 20px rgba(14,165,233,0.4);
    }

    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        font-weight: 700;
        border: none;
        border-radius: 8px;
    }

    .stSelectbox label, .stMultiSelect label, .stSlider label, .stNumberInput label {
        color: #94a3b8;
        font-weight: 600;
        font-size: 13px;
    }

    h1 { font-weight: 800; letter-spacing: -0.03em; color: #f8fafc; }
    h2 { font-weight: 700; letter-spacing: -0.02em; color: #f1f5f9; }
    h3 { font-weight: 600; color: #e2e8f0; }
</style>
"""


import base64
import os

def get_logo_base64():
    """Load and base64 encode logo image from assets directory."""
    logo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "logo.jpeg"))
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
            return f"data:image/jpeg;base64,{encoded}"
    return None


def init_session_state():
    """Ensure global session state variables are initialized across all pages."""
    from utils.data_loader import load_default_sample
    import pandas as pd

    if "df_raw" not in st.session_state or st.session_state.df_raw is None:
        st.session_state.df_raw = load_default_sample()

    if "df_clean" not in st.session_state or st.session_state.df_clean is None:
        st.session_state.df_clean = st.session_state.df_raw.copy()

    if "datetime_col" not in st.session_state:
        st.session_state.datetime_col = "Datetime"

    if "target_col" not in st.session_state:
        cols = st.session_state.df_clean.columns.tolist()
        st.session_state.target_col = (
            "T2M" if "T2M" in cols else
            "PRECTOTCORR" if "PRECTOTCORR" in cols else
            cols[1] if len(cols) > 1 else cols[0]
        )

    if "trained_models" not in st.session_state:
        st.session_state.trained_models = {}

    if "df_train" not in st.session_state:
        n = len(st.session_state.df_clean)
        st.session_state.df_train = st.session_state.df_clean.iloc[:int(n * 0.8)]
        st.session_state.df_test  = st.session_state.df_clean.iloc[int(n * 0.8):]

    if "split_ratio" not in st.session_state:
        st.session_state.split_ratio = 80

    if "selected_features" not in st.session_state:
        df = st.session_state.df_clean
        dt = st.session_state.datetime_col
        st.session_state.selected_features = [
            c for c in df.columns
            if c != dt and pd.api.types.is_numeric_dtype(df[c])
        ]


def render_header(title="ClimaXplore", subtitle="NASA POWER Historical Climate Analytics & Deep Forecasting"):
    """Render consistent top visual header with official logo and custom CSS."""
    init_session_state()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    rows = len(st.session_state.df_clean) if "df_clean" in st.session_state and st.session_state.df_clean is not None else 0
    trained = len(st.session_state.get("trained_models", {}))
    target = st.session_state.get("target_col", "N/A")

    active_label = f"{rows:,} rows · {target}" if rows > 0 else "No Dataset Loaded"
    model_label = f"{trained} model{'s' if trained != 1 else ''} trained" if trained > 0 else "No models yet"

    logo_b64 = get_logo_base64()

    # Render sidebar top branding with official logo
    if logo_b64:
        st.sidebar.markdown(
            f"""
            <div style="padding: 4px 0 12px 0; text-align: center;">
                <img src="{logo_b64}" style="max-width: 100%; height: auto; border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.4);" alt="ClimaXplore Logo" />
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.sidebar.markdown(
            """
            <div style="padding: 4px 0 12px 0;">
                <div style="font-size: 20px; font-weight: 800; color: #f8fafc; display: flex; align-items: center; gap: 8px;">
                    <span>🌍</span> <span>ClimaXplore</span>
                </div>
                <div style="font-size: 11px; color: #64748b; margin-top: 2px;">
                    NASA POWER Historical Analytics
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Render main top visual header with logo image
    logo_html = f'<img src="{logo_b64}" style="height: 52px; border-radius: 6px;" alt="Logo" />' if logo_b64 else '<div class="cx-logo-icon">🌍</div>'

    st.markdown(
        f"""
        <div class="cx-header">
            <div class="cx-logo-box">
                {logo_html}
                <div>
                    <div class="cx-logo-title">{title}</div>
                    <div class="cx-logo-subtitle">{subtitle}</div>
                </div>
            </div>
            <div style="display:flex;gap:10px;align-items:center;">
                <span class="cx-badge">📊 {active_label}</span>
                <span class="cx-badge cx-badge-green">🤖 {model_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_footer():
    """Render persistent custom footer."""
    st.markdown(
        """
        <div class="cx-footer">
            ClimaXplore&nbsp;·&nbsp;NASA POWER Climate Analytics Platform&nbsp;·&nbsp;
            Powered by Streamlit, Plotly &amp; scikit-learn&nbsp;·&nbsp;
            Phase 1: Visualize &nbsp;|&nbsp; Phase 2: Analyze &nbsp;|&nbsp; Phase 3: Predict
        </div>
        <div style="height:48px;"></div>
        """,
        unsafe_allow_html=True
    )


def render_phase_badge(phase: str, icon: str = ""):
    """Render a phase section header badge."""
    st.markdown(
        f"""<div class="cx-phase-header">
            <span class="cx-phase-label">{icon} {phase}</span>
        </div>""",
        unsafe_allow_html=True
    )


def render_metric_cards(metrics: dict):
    """Render a row of metric cards from a dict {label: (value, sub)}."""
    cols = st.columns(len(metrics))
    color_map = {
        "R²": "#4ade80", "MAE": "#fbbf24", "MSE": "#f87171",
        "RMSE": "#fb923c", "MAPE": "#a78bfa"
    }
    for col, (label, (val, sub)) in zip(cols, metrics.items()):
        color = color_map.get(label, "#38bdf8")
        col.markdown(
            f"""<div class="cx-metric-box">
                <div class="cx-metric-label">{label}</div>
                <div class="cx-metric-val" style="color:{color};">{val}</div>
                <div class="cx-metric-sub">{sub}</div>
            </div>""",
            unsafe_allow_html=True
        )


def render_info(msg: str, kind: str = "success"):
    """Render a styled info/warning/error box."""
    icons = {"success": "✅", "warning": "⚠️", "error": "❌"}
    st.markdown(
        f'<div class="cx-info-box {kind}">{icons.get(kind, "")} {msg}</div>',
        unsafe_allow_html=True
    )
