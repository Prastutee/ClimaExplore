"""
Phase 3: Predict — Prediction & Comparison Dashboard
Multi-model side-by-side evaluation, comparative charts, residual analysis, and performance ranking.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from utils.ui import render_header, render_footer

st.set_page_config(page_title="Prediction Comparison — ClimaXplore", page_icon="🏆", layout="wide")
render_header(title="Prediction & Comparison Dashboard",
              subtitle="Multi-Model Side-by-Side Evaluation · Residual Analysis · Performance Rankings")

st.title("🏆 Prediction & Comparison Dashboard")

trained_models = st.session_state.get("trained_models", {})
target_col     = st.session_state.get("target_col", "PRECTOTCORR")

if not trained_models:
    st.warning("No trained models found in session. Please train at least one model on the **Model Training** page first.")
    st.markdown(
        """
        <div class="cx-card">
            <h4 style="color:#fbbf24;">🤖 No models trained yet</h4>
            <p style="color:#94a3b8;">Go to <strong>9 — Model Training</strong> and train at least one model
            (SARIMA, Prophet, Random Forest, XGBoost, or LSTM), then return here for the comparison dashboard.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    render_footer()
    st.stop()

# ── Model Selector ────────────────────────────────────────────────────────────
all_model_names = list(trained_models.keys())
selected_models = st.multiselect(
    "📋 Select Models to Compare",
    all_model_names,
    default=all_model_names,
    help="Choose which trained models to include in the comparison."
)

if not selected_models:
    st.info("Select at least one model above to begin.")
    render_footer()
    st.stop()

models_data = {k: trained_models[k] for k in selected_models}

# ── Build metrics dataframe ───────────────────────────────────────────────────
metrics_rows = []
for name, res in models_data.items():
    m = res["metrics"]
    metrics_rows.append({
        "Model": name,
        "R²":       round(m["R2"], 4),
        "MAE":      round(m["MAE"], 4),
        "MSE":      round(m["MSE"], 4),
        "RMSE":     round(m["RMSE"], 4),
        "MAPE (%)": round(m["MAPE"], 2),
    })
metrics_df = pd.DataFrame(metrics_rows).sort_values("R²", ascending=False).reset_index(drop=True)
best_model  = metrics_df.iloc[0]["Model"]
best_r2     = metrics_df.iloc[0]["R²"]

# ── Hero — Best Model Banner ──────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="cx-card" style="border-color:#fbbf24;background:linear-gradient(135deg,rgba(251,191,36,0.08),rgba(14,165,233,0.05));">
        <div style="display:flex;align-items:center;gap:16px;">
            <div style="font-size:40px;">🥇</div>
            <div>
                <div style="font-size:20px;font-weight:800;color:#fbbf24;">Best Model: {best_model}</div>
                <div style="color:#94a3b8;font-size:14px;">R² = {best_r2:.4f} &nbsp;|&nbsp;
                Compared {len(selected_models)} model{'s' if len(selected_models)!=1 else ''} &nbsp;|&nbsp;
                Target: {target_col}</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_rank, tab_overlay, tab_resid, tab_metrics_chart = st.tabs([
    "🏅 Rankings", "📈 Prediction Overlay", "📊 Residual Analysis", "📉 Metric Charts"
])

# ── Rankings ──────────────────────────────────────────────────────────────────
with tab_rank:
    st.subheader("📊 Performance Ranking Table")

    # Medal emojis
    medals = ["🥇", "🥈", "🥉"] + ["" for _ in range(len(metrics_df) - 3)]
    metrics_display = metrics_df.copy()
    metrics_display.insert(0, "Rank", [f"{medals[i]} #{i+1}" for i in range(len(metrics_df))])

    # Color-code best row
    def highlight_best(row):
        styles = []
        for col in row.index:
            if row["Model"] == best_model:
                styles.append("background-color: rgba(251,191,36,0.12); color: #fbbf24; font-weight:700;")
            else:
                styles.append("")
        return styles

    st.dataframe(
        metrics_display.style.apply(highlight_best, axis=1)
                              .background_gradient(subset=["R²"], cmap="Greens")
                              .background_gradient(subset=["MAE", "RMSE"], cmap="Reds_r"),
        use_container_width=True,
        hide_index=True
    )

    # Podium cards
    st.markdown("#### 🏆 Top 3 Podium")
    podium_cols = st.columns(min(3, len(metrics_df)))
    podium_info = [
        ("🥇", "#fbbf24", "1st Place"),
        ("🥈", "#94a3b8", "2nd Place"),
        ("🥉", "#b45309", "3rd Place")
    ]
    for i, (col, (trophy, color, label)) in enumerate(zip(podium_cols, podium_info)):
        if i < len(metrics_df):
            row = metrics_df.iloc[i]
            col.markdown(
                f"""<div class="cx-metric-box" style="border-color:{color}50;">
                    <div style="font-size:28px;">{trophy}</div>
                    <div class="cx-metric-val" style="font-size:16px;color:{color};">{row['Model']}</div>
                    <div class="cx-metric-label" style="margin-top:8px;">{label}</div>
                    <div style="color:#38bdf8;font-weight:700;margin-top:4px;">R² = {row['R²']:.4f}</div>
                    <div style="color:#94a3b8;font-size:11px;">MAE = {row['MAE']:.4f} | RMSE = {row['RMSE']:.4f}</div>
                </div>""",
                unsafe_allow_html=True
            )


# ── Prediction Overlay ────────────────────────────────────────────────────────
with tab_overlay:
    st.subheader("📈 Multi-Model Prediction Overlay")

    # Find common test length
    test_lengths = [len(models_data[m]["y_true"]) for m in selected_models]
    min_len = min(test_lengths)

    colors_palette = [
        "#38bdf8", "#fbbf24", "#4ade80", "#f87171", "#a78bfa",
        "#fb923c", "#34d399", "#60a5fa", "#e879f9", "#facc15"
    ]

    fig_overlay = go.Figure()

    # Actual (from best model)
    best_res = models_data[best_model]
    fig_overlay.add_trace(go.Scatter(
        y=best_res["y_true"][:min_len],
        name="Actual Values",
        line=dict(color="#f8fafc", width=2.5),
        mode="lines"
    ))

    for i, (name, res) in enumerate(models_data.items()):
        color = colors_palette[i % len(colors_palette)]
        fig_overlay.add_trace(go.Scatter(
            y=res["y_pred"][:min_len],
            name=name,
            line=dict(color=color, width=2, dash="dash" if i > 0 else "solid"),
            mode="lines",
            opacity=0.9
        ))

    fig_overlay.update_layout(
        title=f"All Models — Actual vs. Predicted ({target_col})",
        template="plotly_dark",
        xaxis_title="Test Step",
        yaxis_title=target_col,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=500
    )
    st.plotly_chart(fig_overlay, use_container_width=True)

    # Individual overlays in grid
    if len(selected_models) > 1:
        st.markdown("#### Individual Model Overlays")
        n_per_row = 2
        model_list = list(models_data.items())
        for row_i in range(0, len(model_list), n_per_row):
            chunk = model_list[row_i:row_i + n_per_row]
            cols = st.columns(len(chunk))
            for col_w, (name, res) in zip(cols, chunk):
                with col_w:
                    fig_ind = go.Figure()
                    fig_ind.add_trace(go.Scatter(
                        y=res["y_true"][:min_len], name="Actual",
                        line=dict(color="#f8fafc", width=1.5)
                    ))
                    fig_ind.add_trace(go.Scatter(
                        y=res["y_pred"][:min_len], name="Predicted",
                        line=dict(color="#38bdf8", width=2, dash="dash")
                    ))
                    fig_ind.update_layout(
                        title=name,
                        template="plotly_dark",
                        height=280,
                        showlegend=True,
                        margin=dict(t=40, b=20, l=20, r=20),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_ind, use_container_width=True)


# ── Residual Analysis ─────────────────────────────────────────────────────────
with tab_resid:
    st.subheader("📊 Residual Distribution Analysis")

    col_hist, col_box = st.columns(2)

    with col_hist:
        st.markdown("##### Residual Density Histograms")
        fig_hist = go.Figure()
        for i, (name, res) in enumerate(models_data.items()):
            residuals = np.array(res["y_true"][:min_len]) - np.array(res["y_pred"][:min_len])
            color = colors_palette[i % len(colors_palette)]
            fig_hist.add_trace(go.Histogram(
                x=residuals, name=name,
                opacity=0.6, nbinsx=30,
                marker_color=color,
                histnorm="probability density"
            ))

        fig_hist.add_vline(x=0, line_dash="dash", line_color="#f8fafc", line_width=2)
        fig_hist.update_layout(
            title="Residual Density — All Models",
            template="plotly_dark",
            barmode="overlay",
            xaxis_title="Residual",
            yaxis_title="Density",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_box:
        st.markdown("##### Residual Box Plots")
        fig_box_res = go.Figure()
        for i, (name, res) in enumerate(models_data.items()):
            residuals = np.array(res["y_true"][:min_len]) - np.array(res["y_pred"][:min_len])
            color = colors_palette[i % len(colors_palette)]
            fig_box_res.add_trace(go.Box(
                y=residuals, name=name,
                marker_color=color,
                boxmean="sd", line=dict(width=1.5)
            ))
        fig_box_res.add_hline(y=0, line_dash="dash", line_color="#f8fafc")
        fig_box_res.update_layout(
            title="Residual Spread by Model",
            template="plotly_dark",
            yaxis_title="Residual",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_box_res, use_container_width=True)

    # Residual statistics table
    st.markdown("##### Residual Statistics")
    resid_stats = []
    for name, res in models_data.items():
        residuals = np.array(res["y_true"][:min_len]) - np.array(res["y_pred"][:min_len])
        resid_stats.append({
            "Model": name,
            "Mean Residual":   round(float(np.mean(residuals)), 4),
            "Std Residual":    round(float(np.std(residuals)), 4),
            "Min Residual":    round(float(np.min(residuals)), 4),
            "Max Residual":    round(float(np.max(residuals)), 4),
            "Bias (|Mean|)":   round(float(abs(np.mean(residuals))), 4),
        })
    st.dataframe(pd.DataFrame(resid_stats), use_container_width=True, hide_index=True)


# ── Metric Charts ─────────────────────────────────────────────────────────────
with tab_metrics_chart:
    st.subheader("📉 Metric Bar Charts — Model Comparison")

    for metric in ["R²", "MAE", "RMSE", "MAPE (%)"]:
        colors_bar = [
            "#4ade80" if row["Model"] == best_model else "#38bdf8"
            for _, row in metrics_df.iterrows()
        ]
        fig_m = go.Figure(go.Bar(
            x=metrics_df["Model"],
            y=metrics_df[metric],
            marker_color=colors_bar,
            text=[f"{v:.4f}" for v in metrics_df[metric]],
            textposition="outside"
        ))
        fig_m.update_layout(
            title=f"{metric} — Model Comparison (🟢 = Best)",
            template="plotly_dark",
            yaxis_title=metric,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=300,
            margin=dict(t=50, b=20)
        )
        st.plotly_chart(fig_m, use_container_width=True)

    # Radar chart
    if len(selected_models) >= 2:
        st.markdown("#### 🕸 Radar Chart — Normalized Performance")
        radar_metrics = ["R²", "MAE", "RMSE", "MAPE (%)"]
        norm_df = metrics_df[["Model"] + radar_metrics].copy()
        for col in radar_metrics:
            mn, mx = norm_df[col].min(), norm_df[col].max()
            if col == "R²":
                norm_df[col] = (norm_df[col] - mn) / (mx - mn + 1e-10)
            else:
                # Invert so lower is better → higher on chart
                norm_df[col] = 1 - (norm_df[col] - mn) / (mx - mn + 1e-10)

        fig_radar = go.Figure()
        for i, (_, row) in enumerate(norm_df.iterrows()):
            color = colors_palette[i % len(colors_palette)]
            vals  = [row[m] for m in radar_metrics] + [row[radar_metrics[0]]]  # close loop
            cats  = radar_metrics + [radar_metrics[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=cats,
                fill="toself",
                name=row["Model"],
                line=dict(color=color, width=2),
                opacity=0.6
            ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 1], color="#64748b"),
                angularaxis=dict(color="#94a3b8")
            ),
            template="plotly_dark",
            title="Normalized Performance Radar (higher = better for all axes)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=450
        )
        st.plotly_chart(fig_radar, use_container_width=True)

render_footer()
