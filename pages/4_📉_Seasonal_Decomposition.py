"""
Phase 2: Analyze — Seasonal Decomposition Page
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from utils.ui import render_header, render_footer
from utils.data_loader import df_to_csv_bytes

try:
    from statsmodels.tsa.seasonal import seasonal_decompose, STL
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    from scipy.stats import normaltest
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

st.set_page_config(page_title="Seasonal Decomposition — ClimaXplore", page_icon="📉", layout="wide")
render_header(title="Seasonal Decomposition",
              subtitle="Decompose Series into Observed · Trend · Seasonal · Residual Components")

st.title("📉 Seasonal Decomposition")

df = st.session_state.df_clean
datetime_col = st.session_state.get("datetime_col", "Datetime")
target_col   = st.session_state.get("target_col", "PRECTOTCORR")

if df is None or df.empty:
    st.error("No dataset available. Please upload data on the Data Upload page first.")
    st.stop()

numeric_cols = [c for c in df.columns if c != datetime_col and pd.api.types.is_numeric_dtype(df[c])]

# ── Configuration ─────────────────────────────────────────────────────────────
st.markdown("#### ⚙️ Decomposition Settings")
col1, col2, col3, col4 = st.columns(4)
with col1:
    selected_target = st.selectbox(
        "Target Variable",
        numeric_cols,
        index=numeric_cols.index(target_col) if target_col in numeric_cols else 0
    )
with col2:
    model_type = st.selectbox("Model Type", ["additive", "multiplicative"])
with col3:
    period = st.number_input(
        "Seasonal Period",
        min_value=2, max_value=365, value=7,
        help="7 = weekly, 12 = monthly, 365 = annual (annual requires daily data)"
    )
with col4:
    method = st.selectbox("Decomposition Method", ["Classical", "STL (Robust)"])

# ── Prepare series ────────────────────────────────────────────────────────────
df_ts = df.copy()
if datetime_col in df_ts.columns:
    df_ts = df_ts.sort_values(by=datetime_col)
    df_ts.set_index(datetime_col, inplace=True)

series = df_ts[selected_target].dropna()

if model_type == "multiplicative":
    series = series.clip(lower=0.001)

# ── Decompose ─────────────────────────────────────────────────────────────────
if HAS_STATSMODELS and len(series) >= 2 * period:
    try:
        if method == "STL (Robust)":
            stl_result = STL(series, period=int(period), robust=True).fit()
            observed  = series
            trend     = stl_result.trend
            seasonal  = stl_result.seasonal
            resid     = stl_result.resid
        else:
            result   = seasonal_decompose(series, model=model_type, period=int(period), extrapolate_trend=1)
            observed = result.observed
            trend    = result.trend
            seasonal = result.seasonal
            resid    = result.resid
    except Exception as e:
        st.warning(f"Decomposition failed ({e}), using rolling fallback.")
        trend    = series.rolling(window=int(period), center=True).mean()
        seasonal = series - trend
        resid    = series - trend.fillna(0) - seasonal.fillna(0)
        observed = series
else:
    trend    = series.rolling(window=int(period), center=True).mean()
    seasonal = series - trend
    resid    = series - trend.fillna(0) - seasonal.fillna(0)
    observed = series
    st.warning(f"statsmodels not available or series too short. Showing rolling window fallback (period={period}).")

# ── Summary Metric Cards ──────────────────────────────────────────────────────
st.markdown("#### 📊 Component Metrics")
c1, c2, c3, c4 = st.columns(4)

trend_clean = trend.dropna()
seasonal_clean = seasonal.dropna()
resid_clean = resid.dropna()

trend_pct = (trend_clean.std() / observed.std() * 100) if observed.std() > 0 else 0
seasonal_pct = (seasonal_clean.std() / observed.std() * 100) if observed.std() > 0 else 0

c1.metric("📈 Trend Strength", f"{trend_pct:.1f}%", help="Variance explained by trend component")
c2.metric("🌀 Seasonal Strength", f"{seasonal_pct:.1f}%", help="Variance explained by seasonal component")
c3.metric("📉 Residual Std", f"{resid_clean.std():.4f}")
c4.metric("📋 Series Length", f"{len(series):,} obs")

# Normality test on residuals
if HAS_SCIPY and len(resid_clean) > 8:
    _, p_norm = normaltest(resid_clean.dropna())
    normality = "Normal ✓" if p_norm > 0.05 else f"Non-normal (p={p_norm:.4f})"
    st.caption(f"Residual normality test (D'Agostino-Pearson): **{normality}**")

# ── Plotly 4-Panel Chart ──────────────────────────────────────────────────────
st.markdown("#### 📈 Interactive Decomposition Chart")
fig = make_subplots(
    rows=4, cols=1, shared_xaxes=True,
    subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
    row_heights=[0.3, 0.25, 0.25, 0.2],
    vertical_spacing=0.06
)

fig.add_trace(go.Scatter(x=observed.index, y=observed, name="Observed",
                          line=dict(color="#38bdf8", width=1.5)), row=1, col=1)
fig.add_trace(go.Scatter(x=trend_clean.index, y=trend_clean, name="Trend",
                          line=dict(color="#fbbf24", width=2.5)), row=2, col=1)
fig.add_trace(go.Scatter(x=seasonal_clean.index, y=seasonal_clean, name="Seasonal",
                          line=dict(color="#2dd4bf", width=1.5),
                          fill="tozeroy", fillcolor="rgba(45,212,191,0.08)"), row=3, col=1)
fig.add_trace(go.Bar(x=resid_clean.index, y=resid_clean, name="Residual",
                      marker_color="rgba(248,113,113,0.6)"), row=4, col=1)

fig.update_layout(
    height=800,
    template="plotly_dark",
    title_text=f"{method} Decomposition — {selected_target} (period={period})",
    showlegend=True,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", y=1.04)
)
st.plotly_chart(fig, use_container_width=True)

# ── Export Components ─────────────────────────────────────────────────────────
st.markdown("#### 💾 Export Decomposition Components")
export_df = pd.DataFrame({
    "Date":     observed.index,
    "Observed": observed.values,
    "Trend":    trend.reindex(observed.index).values,
    "Seasonal": seasonal.reindex(observed.index).values,
    "Residual": resid.reindex(observed.index).values,
})
st.download_button(
    "📥 Download Decomposition CSV",
    data=df_to_csv_bytes(export_df),
    file_name=f"decomposition_{selected_target}.csv",
    mime="text/csv",
    use_container_width=True
)

render_footer()
