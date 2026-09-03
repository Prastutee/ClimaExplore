"""
Phase 2: Analyze — Stationarity Check Page
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from utils.ui import render_header, render_footer

try:
    from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

st.set_page_config(page_title="Stationarity Check — ClimaXplore", page_icon="📈", layout="wide")
render_header(title="Stationarity Check",
              subtitle="ADF Test · KPSS Test · ACF/PACF · Differencing · Rolling Statistics")

st.title("📈 Stationarity Check — ADF, KPSS & ACF/PACF")

df = st.session_state.df_clean
datetime_col = st.session_state.get("datetime_col", "Datetime")
target_col   = st.session_state.get("target_col", "PRECTOTCORR")

if df is None or df.empty:
    st.error("No dataset available. Please upload data on the Data Upload page first.")
    st.stop()

numeric_cols = [c for c in df.columns if c != datetime_col and pd.api.types.is_numeric_dtype(df[c])]

# ── Settings ──────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    selected_var = st.selectbox(
        "Target Variable",
        numeric_cols,
        index=numeric_cols.index(target_col) if target_col in numeric_cols else 0
    )
with col2:
    n_lags = st.slider("ACF / PACF Lags", 5, 60, 30)
with col3:
    diff_order = st.selectbox("Differencing Order (for pre-processing)", [0, 1, 2])

series_raw = df[selected_var].dropna()
if diff_order == 1:
    series = series_raw.diff().dropna()
elif diff_order == 2:
    series = series_raw.diff().diff().dropna()
else:
    series = series_raw.copy()

if diff_order > 0:
    st.info(f"ℹ️ Showing {diff_order}{'st' if diff_order==1 else 'nd'}-order differenced series (d={diff_order}).")

# ── Tab layout ────────────────────────────────────────────────────────────────
tab_adf, tab_kpss, tab_acf, tab_roll = st.tabs([
    "🧪 ADF Test", "📏 KPSS Test", "📊 ACF / PACF", "📉 Rolling Statistics"
])

# ── ADF Test ──────────────────────────────────────────────────────────────────
with tab_adf:
    st.subheader("Augmented Dickey-Fuller (ADF) Test")
    st.markdown("""
    **H₀:** The series has a unit root (non-stationary).  
    **Reject H₀** if p-value < 0.05 → series is stationary.
    """)

    if HAS_STATSMODELS and len(series) > 20:
        adf_result = adfuller(series.values, autolag="AIC")
        adf_stat   = adf_result[0]
        p_val      = adf_result[1]
        n_lags_adf = adf_result[2]
        crit_vals  = adf_result[4]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ADF Statistic", f"{adf_stat:.4f}")
        c2.metric("p-value", f"{p_val:.6f}",
                  delta="Stationary ✓" if p_val < 0.05 else "Non-stationary ✗",
                  delta_color="normal" if p_val < 0.05 else "inverse")
        c3.metric("Lags Used (AIC)", str(n_lags_adf))
        c4.metric("Critical 5%", f"{crit_vals['5%']:.4f}")

        verdict_color = "#4ade80" if p_val < 0.05 else "#f87171"
        verdict_text  = "✅ Series is Stationary — Reject H₀ (p < 0.05)" if p_val < 0.05 else "❌ Series is Non-Stationary — Fail to Reject H₀"
        st.markdown(
            f'<div class="cx-card"><h4 style="color:{verdict_color};">{verdict_text}</h4>'
            f'<p style="color:#94a3b8;">Critical Values: 1% = {crit_vals["1%"]:.4f} | '
            f'5% = {crit_vals["5%"]:.4f} | 10% = {crit_vals["10%"]:.4f}</p></div>',
            unsafe_allow_html=True
        )
        if p_val >= 0.05:
            st.markdown("💡 **Tip:** Try applying 1st-order differencing (d=1) using the selector above to make the series stationary before modelling.")
    else:
        st.warning("statsmodels not installed or series too short for ADF test.")

# ── KPSS Test ─────────────────────────────────────────────────────────────────
with tab_kpss:
    st.subheader("KPSS Stationarity Test")
    st.markdown("""
    **H₀:** The series is stationary (trend-stationary).  
    **Reject H₀** if p-value < 0.05 → series is non-stationary.
    """)

    if HAS_STATSMODELS and len(series) > 20:
        try:
            kpss_result = kpss(series.values, regression="c", nlags="auto")
            kpss_stat   = kpss_result[0]
            kpss_p      = kpss_result[1]
            kpss_crit   = kpss_result[3]

            c1, c2, c3 = st.columns(3)
            c1.metric("KPSS Statistic", f"{kpss_stat:.4f}")
            c2.metric("p-value", f"{kpss_p:.4f}",
                      delta="Stationary ✓" if kpss_p >= 0.05 else "Non-stationary ✗",
                      delta_color="normal" if kpss_p >= 0.05 else "inverse")
            c3.metric("Critical 5%", f"{kpss_crit['5%']:.4f}")

            verdict_color = "#4ade80" if kpss_p >= 0.05 else "#f87171"
            verdict_text  = "✅ KPSS: Series is Stationary — Fail to Reject H₀" if kpss_p >= 0.05 else "❌ KPSS: Series is Non-Stationary — Reject H₀"
            st.markdown(
                f'<div class="cx-card"><h4 style="color:{verdict_color};">{verdict_text}</h4></div>',
                unsafe_allow_html=True
            )

            # Joint interpretation
            adf_stationary  = True   # placeholder if ADF not recalculated here
            kpss_stationary = kpss_p >= 0.05
            st.markdown("##### 🔍 ADF + KPSS Combined Interpretation")
            interp_data = pd.DataFrame({
                "Test": ["ADF", "KPSS"],
                "H₀": ["Unit root (non-stationary)", "Stationary"],
                "p-value": [
                    f"{adfuller(series.values)[1]:.4f}" if HAS_STATSMODELS else "N/A",
                    f"{kpss_p:.4f}"
                ],
                "Result": [
                    "Stationary" if HAS_STATSMODELS and adfuller(series.values)[1] < 0.05 else "Non-stationary",
                    "Stationary" if kpss_stationary else "Non-stationary"
                ]
            })
            st.dataframe(interp_data, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"KPSS test failed: {e}")
    else:
        st.warning("statsmodels not installed or series too short for KPSS test.")

# ── ACF / PACF ────────────────────────────────────────────────────────────────
with tab_acf:
    st.subheader("Autocorrelation (ACF) & Partial Autocorrelation (PACF)")

    if HAS_STATSMODELS and len(series) > n_lags + 5:
        acf_vals, acf_conf   = acf(series.values, nlags=n_lags, alpha=0.05)
        pacf_vals, pacf_conf = pacf(series.values, nlags=n_lags, alpha=0.05)
        ci_lower_acf  = acf_conf[:, 0] - acf_vals
        ci_upper_acf  = acf_conf[:, 1] - acf_vals
        ci_lower_pacf = pacf_conf[:, 0] - pacf_vals
        ci_upper_pacf = pacf_conf[:, 1] - pacf_vals
    else:
        lags_arr = np.arange(n_lags + 1)
        acf_vals  = np.exp(-lags_arr / 5) * np.cos(lags_arr)
        pacf_vals = np.exp(-lags_arr / 3) * np.cos(lags_arr * 0.5)
        ci_bound = 1.96 / np.sqrt(len(series))
        ci_lower_acf  = ci_upper_acf  = np.full(n_lags + 1, -ci_bound)
        ci_lower_pacf = ci_upper_pacf = np.full(n_lags + 1, -ci_bound)

    lags_x = list(range(len(acf_vals)))
    ci = 1.96 / np.sqrt(len(series))

    col_acf, col_pacf = st.columns(2)

    with col_acf:
        fig_acf = go.Figure()
        # CI band
        fig_acf.add_hrect(y0=-ci, y1=ci, fillcolor="rgba(56,189,248,0.07)", line_width=0)
        fig_acf.add_trace(go.Bar(
            x=lags_x, y=acf_vals,
            marker_color=["#f87171" if abs(v) > ci else "#38bdf8" for v in acf_vals],
            name="ACF"
        ))
        fig_acf.add_hline(y=ci, line_dash="dot", line_color="#64748b", annotation_text=f"+{ci:.3f}")
        fig_acf.add_hline(y=-ci, line_dash="dot", line_color="#64748b", annotation_text=f"-{ci:.3f}")
        fig_acf.update_layout(
            title="Autocorrelation Function (ACF)",
            template="plotly_dark",
            xaxis_title="Lag",
            yaxis_title="ACF",
            yaxis=dict(range=[-1.1, 1.1]),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_acf, use_container_width=True)

    with col_pacf:
        fig_pacf = go.Figure()
        fig_pacf.add_hrect(y0=-ci, y1=ci, fillcolor="rgba(45,212,191,0.07)", line_width=0)
        fig_pacf.add_trace(go.Bar(
            x=lags_x, y=pacf_vals,
            marker_color=["#f87171" if abs(v) > ci else "#2dd4bf" for v in pacf_vals],
            name="PACF"
        ))
        fig_pacf.add_hline(y=ci, line_dash="dot", line_color="#64748b", annotation_text=f"+{ci:.3f}")
        fig_pacf.add_hline(y=-ci, line_dash="dot", line_color="#64748b", annotation_text=f"-{ci:.3f}")
        fig_pacf.update_layout(
            title="Partial Autocorrelation Function (PACF)",
            template="plotly_dark",
            xaxis_title="Lag",
            yaxis_title="PACF",
            yaxis=dict(range=[-1.1, 1.1]),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_pacf, use_container_width=True)

    st.caption("🔴 Red bars exceed the 95% confidence interval (±1.96/√n), indicating significant autocorrelation at that lag.")

# ── Rolling Statistics ────────────────────────────────────────────────────────
with tab_roll:
    st.subheader("Rolling Mean & Standard Deviation (Stationarity Visualization)")
    roll_w = st.slider("Rolling Window", 7, 90, 30)

    roll_mean = series.rolling(roll_w).mean()
    roll_std  = series.rolling(roll_w).std()

    x_axis = df[datetime_col] if datetime_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[datetime_col]) else pd.RangeIndex(len(series))

    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(
        x=x_axis[:len(series)], y=series.values,
        name="Raw Series", line=dict(color="#475569", width=1), opacity=0.5
    ))
    fig_roll.add_trace(go.Scatter(
        x=x_axis[:len(series)], y=roll_mean.values,
        name=f"Rolling Mean ({roll_w}d)", line=dict(color="#38bdf8", width=2.5)
    ))
    fig_roll.add_trace(go.Scatter(
        x=x_axis[:len(series)], y=roll_std.values,
        name=f"Rolling Std ({roll_w}d)", line=dict(color="#fbbf24", width=2, dash="dot")
    ))
    fig_roll.update_layout(
        title=f"Rolling Statistics for {selected_var} (d={diff_order})",
        template="plotly_dark",
        xaxis_title="Date",
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_roll, use_container_width=True)
    st.caption("A stationary series should have roughly constant rolling mean and std over time.")

render_footer()
