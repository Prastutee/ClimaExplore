"""
ClimaXplore — Multi-Page Streamlit Platform Entry Point.

Initializes persistent session state, visual header/logo, footer, and multi-page navigation.
"""

import streamlit as st
from utils.ui import render_header, render_footer
from utils.data_loader import load_default_sample

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ClimaXplore — Climate Analytics Platform",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://power.larc.nasa.gov/",
        "About": "ClimaXplore — NASA POWER Historical Climate Analytics & Deep Forecasting Platform",
    }
)

# ── Initialize Session State ──────────────────────────────────────────────────
if "df_raw" not in st.session_state:
    st.session_state.df_raw = load_default_sample()

if "df_clean" not in st.session_state:
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
    import pandas as pd
    df = st.session_state.df_clean
    dt = st.session_state.datetime_col
    st.session_state.selected_features = [
        c for c in df.columns
        if c != dt and pd.api.types.is_numeric_dtype(df[c])
    ]

# ── Render Landing Page ───────────────────────────────────────────────────────
render_header()

df = st.session_state.df_clean
n_rows = len(df)
n_cols = len(df.columns)
n_models = len(st.session_state.trained_models)
target = st.session_state.target_col

st.markdown(
    f"""
    <div class="cx-hero">
        <div class="cx-hero-title">ClimaXplore 🌍</div>
        <div class="cx-hero-subtitle">
            End-to-end climate analytics & deep forecasting powered by NASA POWER historical data
        </div>
        <div style="display:flex;justify-content:center;gap:16px;flex-wrap:wrap;">
            <span class="cx-badge">📅 {n_rows:,} observations</span>
            <span class="cx-badge cx-badge-green">📊 {n_cols} variables</span>
            <span class="cx-badge cx-badge-amber">🎯 Target: {target}</span>
            <span class="cx-badge cx-badge-red">🤖 {n_models} models trained</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Phase overview cards
st.markdown(
    """
    <div class="cx-feature-grid">
        <div class="cx-feature-card">
            <div class="cx-feature-icon">📊</div>
            <div class="cx-feature-title">Phase 1 — Visualize</div>
            <div class="cx-feature-desc">
                Upload your climate CSV, map the Datetime column, select target variables,
                impute missing values (Mean, Median, Mode, KNN), and explore
                interactive histograms, line charts, and box plots.
            </div>
        </div>
        <div class="cx-feature-card">
            <div class="cx-feature-icon">🔬</div>
            <div class="cx-feature-title">Phase 2 — Analyze</div>
            <div class="cx-feature-desc">
                Run seasonal decomposition (Observed, Trend, Seasonal, Residual),
                correlation heatmaps, feature importance ranking, ADF / KPSS
                stationarity tests with ACF/PACF, and resample/export datasets.
            </div>
        </div>
        <div class="cx-feature-card">
            <div class="cx-feature-icon">🤖</div>
            <div class="cx-feature-title">Phase 3 — Predict</div>
            <div class="cx-feature-desc">
                Train SARIMA, Prophet, Random Forest, XGBoost, and LSTM models.
                Load pre-trained models (.pkl, .joblib, .json). Compare all models
                side-by-side with residual analysis and performance rankings.
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# NASA POWER parameter reference
st.markdown("### 🛰 NASA POWER Parameter Reference")
import pandas as pd
nasa_params = pd.DataFrame({
    "Parameter": ["T2M", "T2M_MIN", "T2M_MAX", "PRECTOTCORR", "WS2M", "RH2M", "ALLSKY_SFC_SW_DW"],
    "Description": [
        "Temperature at 2 Meters",
        "Minimum Temperature at 2 Meters",
        "Maximum Temperature at 2 Meters",
        "Bias-Corrected Precipitation",
        "Wind Speed at 2 Meters",
        "Relative Humidity at 2 Meters",
        "All-Sky Surface Shortwave Downward Irradiance",
    ],
    "Unit": ["°C", "°C", "°C", "mm/day", "m/s", "%", "kW·h/m²/day"],
    "Typical Range": ["−20 to 50", "−25 to 45", "−15 to 55", "0 to 200+", "0 to 30", "0 to 100", "0 to 10"],
})
st.dataframe(nasa_params, use_container_width=True, hide_index=True)

st.info("👈 Use the sidebar navigation to explore each section of the platform.")

render_footer()
