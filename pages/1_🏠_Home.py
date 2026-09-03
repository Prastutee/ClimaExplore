"""
Phase 1: Visualize — Home Page
"""

import pandas as pd
import streamlit as st
from utils.ui import render_header, render_footer

st.set_page_config(page_title="Home — ClimaXplore", page_icon="🏠", layout="wide")
render_header(title="ClimaXplore — Home", subtitle="NASA POWER Historical Climate Analytics Platform")

# ── Live Session Metrics ──────────────────────────────────────────────────────
df = st.session_state.get("df_clean")
dt_col = st.session_state.get("datetime_col", "Datetime")
tgt_col = st.session_state.get("target_col", "N/A")
n_models = len(st.session_state.get("trained_models", {}))

n_rows = len(df) if df is not None else 0
n_numeric = len([c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]) if df is not None else 0
missing_total = int(df.isnull().sum().sum()) if df is not None else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📅 Total Records", f"{n_rows:,}")
c2.metric("📊 Numeric Features", str(n_numeric))
c3.metric("🎯 Target Variable", tgt_col)
c4.metric("⚠️ Missing Values", str(missing_total))
c5.metric("🤖 Trained Models", str(n_models))

st.markdown("---")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="cx-hero">
        <div class="cx-hero-title">NASA POWER Climate Analytics</div>
        <div class="cx-hero-subtitle">
            Upload, explore, analyze, and forecast historical climate data with
            interactive Plotly visualizations and state-of-the-art ML models.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ── Phase workflow table ──────────────────────────────────────────────────────
st.markdown("### 🗺 Platform Workflow")

phases = pd.DataFrame({
    "Phase": ["Phase 1 — Visualize", "Phase 2 — Analyze", "Phase 3 — Predict"],
    "Pages": [
        "Home · Data Upload & Cleaning · EDA",
        "Seasonal Decomp · Correlation · Feature Selection · Stationarity · Resampling",
        "Model Training · Prediction & Comparison Dashboard"
    ],
    "Key Capabilities": [
        "CSV upload, datetime mapping, KNN/Mean/Median/Mode imputation, histograms, box plots, line charts",
        "Trend/Seasonal/Residual decomposition, Pearson/Spearman heatmaps, ADF/KPSS tests, ACF/PACF, data export",
        "SARIMA, Prophet, RF, XGBoost, LSTM training; pre-trained model loading; R², MAE, MAPE metrics; residual plots"
    ],
})
st.dataframe(phases, use_container_width=True, hide_index=True)

# ── NASA POWER Parameters ─────────────────────────────────────────────────────
st.markdown("### 🛰 Supported NASA POWER Parameters")

params_df = pd.DataFrame({
    "Parameter": ["T2M", "T2M_MIN", "T2M_MAX", "PRECTOTCORR", "WS2M", "RH2M", "ALLSKY_SFC_SW_DW", "PS", "QV2M", "T2MDEW"],
    "Full Name": [
        "Temperature at 2m", "Min Temperature at 2m", "Max Temperature at 2m",
        "Corrected Precipitation", "Wind Speed at 2m", "Relative Humidity at 2m",
        "Surface Shortwave Irradiance", "Surface Pressure",
        "Specific Humidity at 2m", "Dew/Frost Point Temperature"
    ],
    "Unit": ["°C", "°C", "°C", "mm/day", "m/s", "%", "kW·h/m²/day", "kPa", "g/kg", "°C"],
    "Use Case": [
        "Temperature trends & heat analysis",
        "Cold spell detection",
        "Extreme heat events",
        "Rainfall & drought monitoring",
        "Wind energy potential",
        "Humidity & comfort analysis",
        "Solar energy potential",
        "Atmospheric pressure analysis",
        "Moisture content analysis",
        "Dew point climatology"
    ]
})
st.dataframe(params_df, use_container_width=True, hide_index=True)

# ── Quick-start guide ─────────────────────────────────────────────────────────
st.markdown("### 🚀 Quick-Start Guide")
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div class="cx-card">
            <h4 style="color:#38bdf8;">Step 1 — Load Data</h4>
            <p style="color:#94a3b8;">Navigate to <strong>📁 Data Upload & Cleaning</strong> to upload a CSV file or use the built-in NASA POWER synthetic demo dataset.</p>
        </div>
        <div class="cx-card">
            <h4 style="color:#38bdf8;">Step 2 — Explore (EDA)</h4>
            <p style="color:#94a3b8;">Use <strong>📊 EDA</strong> for interactive time series charts, rolling averages, histograms, and box plots with date-range filters.</p>
        </div>
        <div class="cx-card">
            <h4 style="color:#38bdf8;">Step 3 — Analyze Patterns</h4>
            <p style="color:#94a3b8;">Run <strong>Seasonal Decomposition</strong>, <strong>Correlation Matrix</strong>, <strong>Feature Selection</strong>, and <strong>Stationarity Checks</strong>.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="cx-card">
            <h4 style="color:#2dd4bf;">Step 4 — Prepare for Modeling</h4>
            <p style="color:#94a3b8;">Use <strong>🔄 Resampling</strong> to aggregate data (Daily/Monthly), configure train-test splits, and export cleaned datasets.</p>
        </div>
        <div class="cx-card">
            <h4 style="color:#2dd4bf;">Step 5 — Train Models</h4>
            <p style="color:#94a3b8;">Go to <strong>🤖 Model Training</strong> and train SARIMA, Prophet, Random Forest, XGBoost, or LSTM. Or upload a pre-trained .pkl/.joblib model.</p>
        </div>
        <div class="cx-card">
            <h4 style="color:#2dd4bf;">Step 6 — Compare & Rank</h4>
            <p style="color:#94a3b8;">Visit the <strong>🏆 Prediction Comparison Dashboard</strong> for multi-model overlays, residual distributions, and performance rankings.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

render_footer()
