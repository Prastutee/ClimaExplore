"""
Phase 3: Predict — Model Training Page
SARIMA · Prophet · Random Forest · XGBoost · LSTM
"""

import io
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from utils.ui import render_header, render_footer
from utils.models import (
    train_sarima_model, train_prophet_model,
    train_random_forest_model, train_xgboost_model, train_lstm_model,
    load_pretrained_model, predict_with_pretrained,
    calculate_metrics, save_model_bytes
)

st.set_page_config(page_title="Model Training — ClimaXplore", page_icon="🤖", layout="wide")
render_header(title="Model Training & Forecasting",
              subtitle="SARIMA · Prophet · Random Forest · XGBoost · LSTM | Pre-trained Model Loading")

st.title("🤖 Time Series Forecasting Models")

df           = st.session_state.df_clean
datetime_col = st.session_state.get("datetime_col", "Datetime")
target_col   = st.session_state.get("target_col", "PRECTOTCORR")
split_ratio  = st.session_state.get("split_ratio", 80)
test_ratio   = 1.0 - split_ratio / 100.0

if df is None or df.empty:
    st.error("No dataset available. Please upload data on the Data Upload page first.")
    st.stop()

numeric_cols = [c for c in df.columns if c != datetime_col and pd.api.types.is_numeric_dtype(df[c])]


# ── Sidebar — Pre-trained Loader ──────────────────────────────────────────────
st.sidebar.header("📂 Pre-trained Model Loader")
uploaded_model = st.sidebar.file_uploader(
    "Upload Model (.pkl, .joblib, .json)",
    type=["pkl", "joblib", "json"],
    help="Load a previously saved model to run inference on the current dataset."
)
if uploaded_model is not None:
    try:
        loaded_model_obj = load_pretrained_model(uploaded_model, uploaded_model.name)
        st.sidebar.success(f"✅ Loaded **{uploaded_model.name}**")
        predictors = [c for c in numeric_cols if c != target_col]
        if predictors and hasattr(loaded_model_obj, "predict"):
            X_inf = df[predictors].dropna()
            y_inf = df.loc[X_inf.index, target_col]
            preds_inf = predict_with_pretrained(loaded_model_obj, X_inf)
            m_inf = calculate_metrics(y_inf.values, preds_inf)
            st.sidebar.metric("Pre-trained R²", f"{m_inf['R2']:.4f}")
            st.session_state.trained_models[f"PreTrained_{uploaded_model.name}"] = {
                "y_true": y_inf.values, "y_pred": preds_inf,
                "metrics": m_inf, "name": f"Pre-trained ({uploaded_model.name})"
            }
        elif isinstance(loaded_model_obj, dict) and "metrics" in loaded_model_obj:
            # Loaded a result dict directly
            model_name = loaded_model_obj.get("name", uploaded_model.name)
            st.session_state.trained_models[model_name] = loaded_model_obj
            st.sidebar.info(f"Stored result dict for '{model_name}'.")
    except Exception as e:
        st.sidebar.error(f"Failed: {e}")

# Helper to render metrics + overlay for any result
def _render_result(res, model_name):
    m = res["metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("R²",   f"{m['R2']:.4f}")
    c2.metric("MAE",  f"{m['MAE']:.4f}")
    c3.metric("MSE",  f"{m['MSE']:.4f}")
    c4.metric("RMSE", f"{m['RMSE']:.4f}")
    c5.metric("MAPE", f"{m['MAPE']:.2f}%")

    col_overlay, col_resid = st.columns(2)
    with col_overlay:
        fig_ov = go.Figure()
        fig_ov.add_trace(go.Scatter(y=res["y_true"], name="Actual",
                                    line=dict(color="#38bdf8", width=2)))
        fig_ov.add_trace(go.Scatter(y=res["y_pred"], name="Predicted",
                                    line=dict(color="#fbbf24", width=2, dash="dash")))
        fig_ov.update_layout(
            title=f"{model_name} — Actual vs. Predicted",
            template="plotly_dark",
            xaxis_title="Test Step",
            yaxis_title=target_col,
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_ov, use_container_width=True)

    with col_resid:
        residuals = np.array(res["y_true"]) - np.array(res["y_pred"])
        fig_res = go.Figure()
        fig_res.add_trace(go.Histogram(
            x=residuals, nbinsx=30,
            marker_color="rgba(167,139,250,0.7)",
            marker_line_color="#a78bfa", marker_line_width=1,
            name="Residuals"
        ))
        fig_res.add_vline(x=0, line_dash="dash", line_color="#f87171")
        fig_res.update_layout(
            title=f"{model_name} — Residual Distribution",
            template="plotly_dark",
            xaxis_title="Residual",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_res, use_container_width=True)

    # Download model result
    try:
        model_bytes = save_model_bytes(res)
        st.download_button(
            f"💾 Download {model_name} Result (.pkl)",
            data=model_bytes,
            file_name=f"climaxplore_{model_name.lower().replace(' ', '_')}.pkl",
            mime="application/octet-stream"
        )
    except Exception:
        pass


# ── Main Model Tabs ────────────────────────────────────────────────────────────
tab_sarima, tab_prophet, tab_rf, tab_xgb, tab_lstm = st.tabs([
    "📈 SARIMA", "🔮 Prophet", "🌲 Random Forest", "⚡ XGBoost", "🧠 LSTM"
])


# ── SARIMA ────────────────────────────────────────────────────────────────────
with tab_sarima:
    st.subheader("SARIMA Model Configuration")
    st.markdown("Seasonal Auto-Regressive Integrated Moving Average — best for stationary or near-stationary time series with clear seasonality.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Non-seasonal (p, d, q)**")
        p = st.number_input("p — AR order", 0, 5, 1, key="sar_p")
        d = st.number_input("d — Differencing", 0, 2, 1, key="sar_d")
        q = st.number_input("q — MA order", 0, 5, 1, key="sar_q")
    with col2:
        st.markdown("**Seasonal (P, D, Q, s)**")
        P = st.number_input("P — Seasonal AR", 0, 5, 1, key="sar_P")
        D = st.number_input("D — Seasonal Diff", 0, 2, 0, key="sar_D")
        Q = st.number_input("Q — Seasonal MA", 0, 5, 1, key="sar_Q")
    with col3:
        st.markdown("**Seasonality**")
        s = st.number_input("s — Period", 1, 365, 7,
                            help="7=weekly, 12=monthly, 365=annual", key="sar_s")
        test_sz_sar = st.slider("Test Size (%)", 10, 40, int(test_ratio * 100), key="sar_ts")

    if st.button("🚀 Train SARIMA Model", use_container_width=True):
        with st.spinner("Training SARIMA…"):
            res = train_sarima_model(df[target_col], p=p, d=d, q=q, P=P, D=D, Q=Q,
                                     s=s, test_size=test_sz_sar/100)
            st.session_state.trained_models["SARIMA"] = res
            st.success("✅ SARIMA trained successfully!")

    if "SARIMA" in st.session_state.trained_models:
        st.markdown("#### Results")
        _render_result(st.session_state.trained_models["SARIMA"], "SARIMA")


# ── Prophet ───────────────────────────────────────────────────────────────────
with tab_prophet:
    st.subheader("Prophet Model Configuration")
    st.markdown("Facebook Prophet — additive/multiplicative decomposition with trend changepoints and seasonality priors.")

    col1, col2 = st.columns(2)
    with col1:
        cp_scale = st.slider("Changepoint Prior Scale", 0.001, 0.5, 0.05, step=0.005, key="proph_cp")
    with col2:
        sea_scale = st.slider("Seasonality Prior Scale", 0.1, 50.0, 10.0, step=0.5, key="proph_sp")

    test_sz_pr = st.slider("Test Size (%)", 10, 40, int(test_ratio * 100), key="proph_ts")

    if st.button("🚀 Train Prophet Model", use_container_width=True):
        with st.spinner("Training Prophet…"):
            res = train_prophet_model(df, datetime_col, target_col,
                                      test_size=test_sz_pr/100,
                                      changepoint_prior_scale=cp_scale,
                                      seasonality_prior_scale=sea_scale)
            st.session_state.trained_models["Prophet"] = res
            st.success("✅ Prophet trained successfully!")

    if "Prophet" in st.session_state.trained_models:
        st.markdown("#### Results")
        _render_result(st.session_state.trained_models["Prophet"], "Prophet")


# ── Random Forest ─────────────────────────────────────────────────────────────
with tab_rf:
    st.subheader("Random Forest Regressor")
    st.markdown("Ensemble of decision trees using Gini importance. Works well with multivariate feature sets without feature scaling.")

    col1, col2, col3 = st.columns(3)
    with col1:
        n_est_rf = st.slider("n_estimators", 10, 400, 100, step=10, key="rf_n")
    with col2:
        max_d_rf = st.slider("max_depth", 2, 50, 10, key="rf_d")
    with col3:
        test_sz_rf = st.slider("Test Size (%)", 10, 40, int(test_ratio * 100), key="rf_ts")

    predictors_rf = st.multiselect(
        "Predictor Features",
        [c for c in numeric_cols if c != target_col],
        default=[c for c in numeric_cols if c != target_col],
        key="rf_feat"
    )

    if st.button("🚀 Train Random Forest", use_container_width=True):
        if not predictors_rf:
            st.warning("Select at least one predictor feature.")
        else:
            with st.spinner("Training Random Forest…"):
                X = df[predictors_rf].dropna()
                y = df.loc[X.index, target_col]
                split_i = int(len(X) * (1 - test_sz_rf / 100))
                res = train_random_forest_model(
                    X.iloc[:split_i], y.iloc[:split_i],
                    X.iloc[split_i:], y.iloc[split_i:],
                    n_estimators=n_est_rf, max_depth=max_d_rf
                )
                st.session_state.trained_models["Random Forest"] = res
                st.success("✅ Random Forest trained!")

    if "Random Forest" in st.session_state.trained_models:
        st.markdown("#### Results")
        _render_result(st.session_state.trained_models["Random Forest"], "Random Forest")
        # Feature importance bar
        res_rf = st.session_state.trained_models["Random Forest"]
        if res_rf.get("feature_importances"):
            imp = pd.Series(res_rf["feature_importances"]).sort_values()
            fig_imp = go.Figure(go.Bar(
                x=imp.values, y=imp.index, orientation="h",
                marker_color="#38bdf8"
            ))
            fig_imp.update_layout(
                title="Feature Importances",
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_imp, use_container_width=True)


# ── XGBoost ───────────────────────────────────────────────────────────────────
with tab_xgb:
    st.subheader("XGBoost Regressor")
    st.markdown("Gradient boosted trees — handles non-linearity and feature interactions with regularisation to prevent overfitting.")

    col1, col2, col3 = st.columns(3)
    with col1:
        n_est_xgb = st.slider("n_estimators", 10, 400, 100, step=10, key="xgb_n")
    with col2:
        lr_xgb = st.number_input("learning_rate", 0.001, 0.5, 0.1, step=0.01, key="xgb_lr")
    with col3:
        max_d_xgb = st.slider("max_depth", 2, 12, 6, key="xgb_d")

    test_sz_xgb = st.slider("Test Size (%)", 10, 40, int(test_ratio * 100), key="xgb_ts")

    predictors_xgb = st.multiselect(
        "Predictor Features",
        [c for c in numeric_cols if c != target_col],
        default=[c for c in numeric_cols if c != target_col],
        key="xgb_feat"
    )

    if st.button("🚀 Train XGBoost", use_container_width=True):
        if not predictors_xgb:
            st.warning("Select at least one predictor feature.")
        else:
            with st.spinner("Training XGBoost…"):
                X = df[predictors_xgb].dropna()
                y = df.loc[X.index, target_col]
                split_i = int(len(X) * (1 - test_sz_xgb / 100))
                res = train_xgboost_model(
                    X.iloc[:split_i], y.iloc[:split_i],
                    X.iloc[split_i:], y.iloc[split_i:],
                    n_estimators=n_est_xgb, learning_rate=lr_xgb, max_depth=max_d_xgb
                )
                st.session_state.trained_models["XGBoost"] = res
                st.success("✅ XGBoost trained!")

    if "XGBoost" in st.session_state.trained_models:
        st.markdown("#### Results")
        _render_result(st.session_state.trained_models["XGBoost"], "XGBoost")


# ── LSTM ──────────────────────────────────────────────────────────────────────
with tab_lstm:
    st.subheader("LSTM Neural Network")
    st.markdown("Long Short-Term Memory (PyTorch) — captures long-range temporal dependencies. Requires PyTorch for full training; falls back to trend extrapolation if unavailable.")

    col1, col2, col3 = st.columns(3)
    with col1:
        epochs_lstm = st.slider("Epochs", 5, 100, 20, step=5, key="lstm_e")
    with col2:
        hidden_lstm = st.slider("Hidden Dimension", 8, 128, 32, step=8, key="lstm_h")
    with col3:
        lookback_lstm = st.slider("Lookback (days)", 7, 90, 30, key="lstm_lb")

    test_sz_lstm = st.slider("Test Size (%)", 10, 40, int(test_ratio * 100), key="lstm_ts")

    try:
        import torch
        st.info(f"🔥 PyTorch {torch.__version__} detected — full LSTM training enabled.")
    except ImportError:
        st.warning("⚠️ PyTorch not installed — LSTM will use trend-extrapolation fallback. Install `torch` for full support.")

    if st.button("🚀 Train LSTM", use_container_width=True):
        with st.spinner(f"Training LSTM for {epochs_lstm} epochs…"):
            res = train_lstm_model(df[target_col], epochs=epochs_lstm,
                                   hidden_dim=hidden_lstm, lookback=lookback_lstm,
                                   test_size=test_sz_lstm/100)
            st.session_state.trained_models["LSTM"] = res
            st.success("✅ LSTM trained!")

    if "LSTM" in st.session_state.trained_models:
        st.markdown("#### Results")
        _render_result(st.session_state.trained_models["LSTM"], "LSTM")


# ── Summary Table ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 All Trained Models — Summary")
if st.session_state.trained_models:
    rows = []
    for name, res in st.session_state.trained_models.items():
        m = res["metrics"]
        rows.append({
            "Model": name,
            "R²":   round(m["R2"], 4),
            "MAE":  round(m["MAE"], 4),
            "MSE":  round(m["MSE"], 4),
            "RMSE": round(m["RMSE"], 4),
            "MAPE (%)": round(m["MAPE"], 2),
        })
    summary_df = pd.DataFrame(rows).sort_values("R²", ascending=False)
    st.dataframe(summary_df.style.background_gradient(subset=["R²"], cmap="Greens")
                                   .background_gradient(subset=["MAE", "RMSE"], cmap="Reds_r"),
                 use_container_width=True, hide_index=True)
    st.info("🏆 Navigate to the **Prediction Comparison Dashboard** (page 10) to compare all models side-by-side.")
else:
    st.info("Train any model above to see the summary table here.")

render_footer()
