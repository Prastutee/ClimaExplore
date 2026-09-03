"""
Phase 2: Analyze — Correlation Matrix Page
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils.ui import render_header, render_footer

try:
    from scipy.stats import pearsonr, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

st.set_page_config(page_title="Correlation Matrix — ClimaXplore", page_icon="🔥", layout="wide")
render_header(title="Correlation Matrix & Heatmaps",
              subtitle="Pearson · Spearman · Kendall · Rolling Correlation · Lag Analysis")

st.title("🔥 Statistical Correlation Analysis")

df = st.session_state.df_clean
datetime_col = st.session_state.get("datetime_col", "Datetime")
target_col   = st.session_state.get("target_col", "PRECTOTCORR")

if df is None or df.empty:
    st.error("No dataset available. Please upload data on the Data Upload page first.")
    st.stop()

numeric_cols = [c for c in df.columns if c != datetime_col and pd.api.types.is_numeric_dtype(df[c])]

if len(numeric_cols) < 2:
    st.warning("Need at least 2 numeric columns for correlation analysis.")
    st.stop()

# ── Settings ──────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    corr_method = st.selectbox("Correlation Method", ["pearson", "spearman", "kendall"])
with col2:
    selected_target = st.selectbox(
        "Anchor Variable (for bar chart)",
        numeric_cols,
        index=numeric_cols.index(target_col) if target_col in numeric_cols else 0
    )
with col3:
    var_subset = st.multiselect("Variables to Include", numeric_cols, default=numeric_cols)

if len(var_subset) < 2:
    st.info("Select at least 2 variables to compute correlations.")
    st.stop()

corr_matrix = df[var_subset].corr(method=corr_method)

# ── Tab Layout ─────────────────────────────────────────────────────────────────
tab_heat, tab_bar, tab_scatter, tab_roll, tab_lag = st.tabs([
    "🟥 Heatmap", "📊 Bar Chart", "🔵 Scatter Plot", "📈 Rolling Correlation", "⏱ Lag Analysis"
])

# ── Heatmap ───────────────────────────────────────────────────────────────────
with tab_heat:
    st.subheader(f"Inter-Variable Correlation Heatmap ({corr_method.capitalize()})")
    fig_hmap = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title=f"{corr_method.capitalize()} Correlation Matrix",
        template="plotly_dark",
        aspect="auto"
    )
    fig_hmap.update_traces(textfont=dict(size=11))
    fig_hmap.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=500
    )
    st.plotly_chart(fig_hmap, use_container_width=True)

    # Show strong correlations table
    threshold = st.slider("Show pairs with |r| ≥", 0.0, 1.0, 0.5, step=0.05)
    strong_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            r = corr_matrix.iloc[i, j]
            if abs(r) >= threshold:
                strong_pairs.append({
                    "Variable A": corr_matrix.columns[i],
                    "Variable B": corr_matrix.columns[j],
                    "Correlation": round(r, 4),
                    "Strength": "Strong" if abs(r) >= 0.7 else "Moderate"
                })
    if strong_pairs:
        st.dataframe(pd.DataFrame(strong_pairs).sort_values("Correlation", key=abs, ascending=False),
                     use_container_width=True, hide_index=True)
    else:
        st.info(f"No variable pairs with |r| ≥ {threshold}")

# ── Bar Chart ─────────────────────────────────────────────────────────────────
with tab_bar:
    st.subheader(f"Feature Correlation Strength vs. {selected_target}")
    if selected_target in var_subset:
        target_corr = corr_matrix[selected_target].drop(selected_target).sort_values()
        colors = ["#f87171" if v < 0 else "#4ade80" for v in target_corr.values]
        fig_bar = go.Figure(go.Bar(
            x=target_corr.values,
            y=target_corr.index,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in target_corr.values],
            textposition="outside"
        ))
        fig_bar.update_layout(
            title=f"Correlation with {selected_target} ({corr_method})",
            template="plotly_dark",
            xaxis_title="Correlation Coefficient",
            xaxis_range=[-1.1, 1.1],
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info(f"'{selected_target}' not in selected variable subset.")

# ── Scatter Plot ──────────────────────────────────────────────────────────────
with tab_scatter:
    st.subheader("Scatter Plot — Variable Pair")
    col_x, col_y = st.columns(2)
    with col_x:
        x_var = st.selectbox("X Variable", var_subset, index=0)
    with col_y:
        y_var = st.selectbox("Y Variable", var_subset, index=min(1, len(var_subset) - 1))

    if x_var != y_var:
        color_by = None
        if datetime_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
            df_scatter = df[[x_var, y_var, datetime_col]].dropna()
            df_scatter["Year"] = df_scatter[datetime_col].dt.year.astype(str)
            color_by = "Year"
        else:
            df_scatter = df[[x_var, y_var]].dropna()

        fig_sc = px.scatter(
            df_scatter, x=x_var, y=y_var,
            color=color_by,
            trendline="ols",
            title=f"{x_var} vs {y_var}",
            template="plotly_dark",
            opacity=0.6
        )
        fig_sc.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        # Correlation stat
        clean = df[[x_var, y_var]].dropna()
        r_val = corr_matrix.loc[x_var, y_var] if (x_var in corr_matrix and y_var in corr_matrix) else 0
        st.metric(f"{corr_method.capitalize()} r ({x_var} vs {y_var})", f"{r_val:.4f}")
    else:
        st.info("Select two different variables.")

# ── Rolling Correlation ───────────────────────────────────────────────────────
with tab_roll:
    st.subheader("Rolling Correlation Over Time")
    col_rv1, col_rv2, col_rv3 = st.columns(3)
    with col_rv1:
        roll_x = st.selectbox("Variable A", var_subset, index=0, key="rx")
    with col_rv2:
        roll_y = st.selectbox("Variable B", var_subset, index=min(1, len(var_subset)-1), key="ry")
    with col_rv3:
        roll_win = st.slider("Rolling Window (days)", 14, 180, 30, key="rw")

    if roll_x != roll_y and datetime_col in df.columns:
        df_roll = df[[datetime_col, roll_x, roll_y]].dropna().sort_values(datetime_col)
        rolling_corr = df_roll[roll_x].rolling(roll_win).corr(df_roll[roll_y])

        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(
            x=df_roll[datetime_col], y=rolling_corr,
            name=f"Rolling r ({roll_win}d)",
            line=dict(color="#a78bfa", width=2),
            fill="tozeroy",
            fillcolor="rgba(167,139,250,0.1)"
        ))
        fig_roll.add_hline(y=0, line_dash="dash", line_color="#64748b")
        fig_roll.add_hline(y=0.5, line_dash="dot", line_color="#4ade80", annotation_text="r=0.5")
        fig_roll.add_hline(y=-0.5, line_dash="dot", line_color="#f87171", annotation_text="r=-0.5")
        fig_roll.update_layout(
            title=f"Rolling {roll_win}-day Correlation: {roll_x} ↔ {roll_y}",
            template="plotly_dark",
            yaxis=dict(range=[-1.1, 1.1], title="Pearson r"),
            xaxis_title="Date",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_roll, use_container_width=True)
    else:
        st.info("Select two different variables and ensure datetime column is present.")

# ── Lag Correlation ───────────────────────────────────────────────────────────
with tab_lag:
    st.subheader("Lag Cross-Correlation Analysis")
    col_lx, col_ly = st.columns(2)
    with col_lx:
        lag_x = st.selectbox("Lead Variable", var_subset, index=0, key="lx")
    with col_ly:
        lag_y = st.selectbox("Lagged Variable", var_subset, index=min(1, len(var_subset)-1), key="ly")

    max_lag = st.slider("Max Lag (days)", 5, 60, 20)

    if lag_x != lag_y:
        lags = range(-max_lag, max_lag + 1)
        corrs = []
        s1 = df[lag_x].dropna()
        s2 = df[lag_y].dropna()
        min_len = min(len(s1), len(s2))
        s1, s2 = s1.iloc[:min_len].values, s2.iloc[:min_len].values

        for lag in lags:
            if lag < 0:
                r = float(np.corrcoef(s1[:min_len+lag], s2[-lag:])[0, 1])
            elif lag > 0:
                r = float(np.corrcoef(s1[lag:], s2[:min_len-lag])[0, 1])
            else:
                r = float(np.corrcoef(s1, s2)[0, 1])
            corrs.append(r)

        ci = 1.96 / np.sqrt(min_len)
        fig_lag = go.Figure()
        fig_lag.add_trace(go.Bar(
            x=list(lags), y=corrs,
            marker=dict(
                color=corrs,
                colorscale="RdBu_r",
                cmin=-1, cmax=1,
                showscale=True
            ),
            name="Cross-correlation"
        ))
        fig_lag.add_hline(y=ci, line_dash="dot", line_color="#4ade80", annotation_text=f"+95% CI ({ci:.3f})")
        fig_lag.add_hline(y=-ci, line_dash="dot", line_color="#f87171", annotation_text=f"-95% CI ({-ci:.3f})")
        fig_lag.update_layout(
            title=f"Cross-Correlation: {lag_x} ↔ {lag_y} (lags ±{max_lag})",
            template="plotly_dark",
            xaxis_title=f"Lag (positive = {lag_y} lags {lag_x})",
            yaxis_title="Correlation",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_lag, use_container_width=True)
    else:
        st.info("Select two different variables.")

render_footer()
