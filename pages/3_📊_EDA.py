"""
Phase 1: Visualize — Exploratory Data Analysis (EDA) Page
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from utils.ui import render_header, render_footer

st.set_page_config(page_title="EDA — ClimaXplore", page_icon="📊", layout="wide")
render_header(title="Exploratory Data Analysis (EDA)",
              subtitle="Interactive Filters · Line Charts · Histograms · Box Plots · Scatter Matrix")

st.title("📊 Exploratory Data Analysis")

df = st.session_state.df_clean
datetime_col = st.session_state.get("datetime_col", "Datetime")

if df is None or df.empty:
    st.error("No dataset available. Please upload data on the Data Upload page first.")
    st.stop()

numeric_cols = [c for c in df.columns if c != datetime_col and pd.api.types.is_numeric_dtype(df[c])]

# ── Sidebar Filters ────────────────────────────────────────────────────────────
st.sidebar.header("🎛 Filter Controls")
selected_vars = st.sidebar.multiselect(
    "Variables to Analyse", numeric_cols,
    default=numeric_cols[:min(4, len(numeric_cols))]
)

# Date-range filter
df_filtered = df.copy()
if datetime_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
    min_date = df[datetime_col].min().date()
    max_date = df[datetime_col].max().date()
    date_range = st.sidebar.date_input("Date Range", [min_date, max_date])
    if len(date_range) == 2:
        df_filtered = df[
            (df[datetime_col].dt.date >= date_range[0]) &
            (df[datetime_col].dt.date <= date_range[1])
        ]

rolling_win = st.sidebar.slider("Rolling Average Window (days)", 3, 90, 7)

if not selected_vars:
    st.warning("Please select at least one variable from the sidebar.")
    st.stop()

# ── Top-level KPI row ─────────────────────────────────────────────────────────
st.markdown("#### 📋 Summary Statistics")
stats = df_filtered[selected_vars].describe().T[["mean", "std", "min", "max"]]
stats.columns = ["Mean", "Std Dev", "Min", "Max"]
st.dataframe(stats.style.background_gradient(cmap="Blues", axis=0), use_container_width=True)

st.markdown("---")

# ── Tab Layout ─────────────────────────────────────────────────────────────────
tab_line, tab_hist, tab_box, tab_scatter, tab_corr = st.tabs([
    "📈 Line Chart", "📊 Histograms", "📦 Box Plots", "🔵 Scatter Matrix", "🔥 Quick Correlation"
])

# ── Line Chart ─────────────────────────────────────────────────────────────────
with tab_line:
    st.subheader("Time Series with Rolling Average Overlay")
    has_dt = datetime_col in df_filtered.columns and pd.api.types.is_datetime64_any_dtype(df_filtered[datetime_col])

    if has_dt and selected_vars:
        fig_line = go.Figure()
        colors = px.colors.qualitative.Safe

        for i, col in enumerate(selected_vars):
            color = colors[i % len(colors)]
            fig_line.add_trace(go.Scatter(
                x=df_filtered[datetime_col], y=df_filtered[col],
                name=col, mode="lines",
                line=dict(color=color, width=1.5),
                opacity=0.7
            ))
            rolling = df_filtered[col].rolling(window=rolling_win, center=True).mean()
            fig_line.add_trace(go.Scatter(
                x=df_filtered[datetime_col], y=rolling,
                name=f"{col} ({rolling_win}d avg)", mode="lines",
                line=dict(color=color, width=2.5, dash="dot"),
                opacity=1.0
            ))

        fig_line.update_layout(
            title=f"Climate Parameter Trends — {rolling_win}-day Rolling Average",
            template="plotly_dark",
            xaxis_title="Date",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Datetime column not found or no variables selected.")

# ── Histograms ─────────────────────────────────────────────────────────────────
with tab_hist:
    st.subheader("Distribution Histograms with KDE")
    n_cols_grid = 2
    var_chunks = [selected_vars[i:i+n_cols_grid] for i in range(0, len(selected_vars), n_cols_grid)]

    for chunk in var_chunks:
        cols_row = st.columns(len(chunk))
        for col_w, var in zip(cols_row, chunk):
            with col_w:
                fig_h = go.Figure()
                data = df_filtered[var].dropna()
                fig_h.add_trace(go.Histogram(
                    x=data, nbinsx=40,
                    name=var,
                    marker_color="rgba(56,189,248,0.7)",
                    marker_line_color="#38bdf8",
                    marker_line_width=1
                ))
                # KDE overlay
                from scipy.stats import gaussian_kde
                try:
                    kde = gaussian_kde(data)
                    x_range = np.linspace(data.min(), data.max(), 200)
                    kde_vals = kde(x_range) * len(data) * (data.max() - data.min()) / 40
                    fig_h.add_trace(go.Scatter(
                        x=x_range, y=kde_vals, mode="lines",
                        name="KDE", line=dict(color="#fbbf24", width=2)
                    ))
                except Exception:
                    pass

                fig_h.update_layout(
                    title=f"{var} Distribution",
                    template="plotly_dark",
                    showlegend=True,
                    margin=dict(t=40, b=20),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_h, use_container_width=True)

# ── Box Plots ─────────────────────────────────────────────────────────────────
with tab_box:
    st.subheader("Box Plots — Outlier & Distribution Analysis")
    fig_box = go.Figure()
    colors_box = px.colors.qualitative.Pastel
    for i, var in enumerate(selected_vars):
        fig_box.add_trace(go.Box(
            y=df_filtered[var].dropna(),
            name=var,
            marker_color=colors_box[i % len(colors_box)],
            boxmean="sd",
            line=dict(width=2)
        ))

    fig_box.update_layout(
        title="Box Plots for Selected Climate Variables",
        template="plotly_dark",
        yaxis_title="Value",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True
    )
    st.plotly_chart(fig_box, use_container_width=True)

    # Monthly box plots for the primary target
    target_col = st.session_state.get("target_col", selected_vars[0])
    if target_col in df_filtered.columns and datetime_col in df_filtered.columns:
        st.subheader(f"Monthly Distribution — {target_col}")
        df_month = df_filtered[[datetime_col, target_col]].copy().dropna()
        if pd.api.types.is_datetime64_any_dtype(df_month[datetime_col]):
            df_month["Month"] = df_month[datetime_col].dt.strftime("%b")
            df_month["MonthNum"] = df_month[datetime_col].dt.month
            df_month_sorted = df_month.sort_values("MonthNum")
            fig_monthly = px.box(
                df_month_sorted, x="Month", y=target_col,
                title=f"Monthly Distribution of {target_col}",
                template="plotly_dark",
                color="Month",
                category_orders={"Month": ["Jan","Feb","Mar","Apr","May","Jun",
                                            "Jul","Aug","Sep","Oct","Nov","Dec"]}
            )
            fig_monthly.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False
            )
            st.plotly_chart(fig_monthly, use_container_width=True)

# ── Scatter Matrix ─────────────────────────────────────────────────────────────
with tab_scatter:
    st.subheader("Scatter Matrix (Pair Plot)")
    if len(selected_vars) < 2:
        st.info("Select at least 2 variables from the sidebar to view the scatter matrix.")
    elif len(selected_vars) > 6:
        st.warning("Showing scatter matrix for first 6 selected variables to maintain performance.")
        plot_vars = selected_vars[:6]
    else:
        plot_vars = selected_vars

    if len(selected_vars) >= 2:
        fig_scatter = px.scatter_matrix(
            df_filtered[plot_vars].dropna(),
            dimensions=plot_vars,
            title="Scatter Matrix — Feature Pair Relationships",
            template="plotly_dark",
            color_discrete_sequence=["#38bdf8"],
            opacity=0.5
        )
        fig_scatter.update_traces(marker=dict(size=3))
        fig_scatter.update_layout(
            height=650,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# ── Quick Correlation ─────────────────────────────────────────────────────────
with tab_corr:
    st.subheader("Quick Correlation Heatmap")
    if len(selected_vars) >= 2:
        corr_m = df_filtered[selected_vars].corr()
        fig_corr = px.imshow(
            corr_m,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Pearson Correlation Matrix",
            template="plotly_dark",
            zmin=-1, zmax=1
        )
        fig_corr.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # Bar chart vs target
        target = st.session_state.get("target_col", selected_vars[0])
        if target in selected_vars:
            st.subheader(f"Correlation Strength vs. {target}")
            target_corr = corr_m[target].drop(target).sort_values()
            fig_bar = go.Figure(go.Bar(
                x=target_corr.values,
                y=target_corr.index,
                orientation="h",
                marker=dict(
                    color=target_corr.values,
                    colorscale="RdBu_r",
                    cmin=-1, cmax=1,
                    showscale=True
                )
            ))
            fig_bar.update_layout(
                title=f"Feature Correlation with {target}",
                template="plotly_dark",
                xaxis_title="Pearson r",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Select at least 2 variables to compute correlations.")

render_footer()
