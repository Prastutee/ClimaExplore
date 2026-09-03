"""
Phase 2: Analyze — Feature Selection Page
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score
from utils.ui import render_header, render_footer

st.set_page_config(page_title="Feature Selection — ClimaXplore", page_icon="🎯", layout="wide")
render_header(title="Feature Selection & Importance",
              subtitle="Random Forest Importance · Mutual Information · Cumulative Threshold · R² Validation")

st.title("🎯 Feature Selection & Importance Metrics")

df = st.session_state.df_clean
datetime_col = st.session_state.get("datetime_col", "Datetime")
target_col   = st.session_state.get("target_col", "PRECTOTCORR")

if df is None or df.empty:
    st.error("No dataset available. Please upload data on the Data Upload page first.")
    st.stop()

numeric_cols = [c for c in df.columns if c != datetime_col and pd.api.types.is_numeric_dtype(df[c])]

if len(numeric_cols) < 2:
    st.warning("Feature selection requires at least 2 numeric columns.")
    st.stop()

# ── Configuration ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    selected_target = st.selectbox(
        "🎯 Target Variable",
        numeric_cols,
        index=numeric_cols.index(target_col) if target_col in numeric_cols else 0
    )
with col2:
    n_estimators = st.slider("Random Forest n_estimators", 50, 300, 100, step=25)

predictor_cols = [c for c in numeric_cols if c != selected_target]
X_raw = df[predictor_cols].dropna()
y_raw = df.loc[X_raw.index, selected_target].dropna()
X_raw = X_raw.loc[y_raw.index]

# ── Compute Importance ────────────────────────────────────────────────────────
with st.spinner("Computing feature importance scores…"):
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X_raw, y_raw)
    rf_imp = pd.Series(rf.feature_importances_, index=predictor_cols)

    mi_scores = mutual_info_regression(X_raw, y_raw, random_state=42)
    mi_imp = pd.Series(mi_scores, index=predictor_cols)

    corr_imp = pd.Series(
        [abs(df[c].corr(df[selected_target])) for c in predictor_cols],
        index=predictor_cols
    )

# Normalize all to [0, 1] for combined view
def normalize(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn + 1e-10)

rf_n   = normalize(rf_imp)
mi_n   = normalize(mi_imp)
corr_n = normalize(corr_imp)
combined = (rf_n + mi_n + corr_n) / 3
combined_sorted = combined.sort_values(ascending=False)

# ── Combined Ranking Table ─────────────────────────────────────────────────────
st.markdown("#### 🏅 Combined Feature Ranking")
ranking_df = pd.DataFrame({
    "Feature":           combined_sorted.index,
    "RF Importance":     rf_imp[combined_sorted.index].round(4),
    "Mutual Info":       mi_imp[combined_sorted.index].round(4),
    "Pearson |r|":       corr_imp[combined_sorted.index].round(4),
    "Combined Score":    combined_sorted.round(4),
    "Rank":              range(1, len(combined_sorted) + 1)
}).reset_index(drop=True)
st.dataframe(ranking_df, use_container_width=True, hide_index=True)

st.markdown("---")
tab_rf, tab_mi, tab_cumul, tab_val = st.tabs([
    "🌲 RF Importance", "💡 Mutual Info", "📈 Cumulative", "✅ Validation R²"
])

# ── RF Importance ─────────────────────────────────────────────────────────────
with tab_rf:
    st.subheader("Random Forest Feature Importance (Gini)")
    rf_sorted = rf_imp.sort_values()
    fig_rf = go.Figure(go.Bar(
        x=rf_sorted.values,
        y=rf_sorted.index,
        orientation="h",
        marker=dict(
            color=rf_sorted.values,
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Gini Imp.")
        ),
        text=[f"{v:.4f}" for v in rf_sorted.values],
        textposition="outside"
    ))
    fig_rf.update_layout(
        title=f"RF Importance — Target: {selected_target}",
        template="plotly_dark",
        xaxis_title="Gini Importance",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_rf, use_container_width=True)

# ── Mutual Information ────────────────────────────────────────────────────────
with tab_mi:
    st.subheader("Mutual Information Regression Scores")
    mi_sorted = mi_imp.sort_values()
    fig_mi = go.Figure(go.Bar(
        x=mi_sorted.values,
        y=mi_sorted.index,
        orientation="h",
        marker=dict(color="#2dd4bf"),
        text=[f"{v:.4f}" for v in mi_sorted.values],
        textposition="outside"
    ))
    fig_mi.update_layout(
        title=f"Mutual Information — Target: {selected_target}",
        template="plotly_dark",
        xaxis_title="MI Score (bits)",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_mi, use_container_width=True)

# ── Cumulative Importance ─────────────────────────────────────────────────────
with tab_cumul:
    st.subheader("Cumulative Importance Curve (RF)")
    rf_desc = rf_imp.sort_values(ascending=False)
    cumul = rf_desc.cumsum() / rf_desc.sum() * 100

    threshold_pct = st.slider("Select Cumulative Importance Threshold (%)", 50, 99, 80)
    n_needed = int((cumul <= threshold_pct).sum()) + 1

    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=list(range(1, len(cumul)+1)),
        y=cumul.values,
        mode="lines+markers",
        name="Cumulative Importance",
        line=dict(color="#38bdf8", width=2.5),
        marker=dict(size=8)
    ))
    fig_cum.add_hline(y=threshold_pct, line_dash="dash", line_color="#fbbf24",
                      annotation_text=f"{threshold_pct}% threshold — {n_needed} features")
    fig_cum.update_layout(
        title=f"Cumulative RF Importance — {n_needed} features reach {threshold_pct}%",
        template="plotly_dark",
        xaxis_title="Number of Features",
        yaxis_title="Cumulative Importance (%)",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    auto_features = rf_desc.index[:n_needed].tolist()
    st.info(f"✅ Auto-selected top-{n_needed} features: **{', '.join(auto_features)}**")

# ── Validation R² ─────────────────────────────────────────────────────────────
with tab_val:
    st.subheader("Cross-Validated R² — Selected vs. All Features")
    manual_sel = st.multiselect(
        "Choose Feature Subset to Validate",
        predictor_cols,
        default=predictor_cols[:min(3, len(predictor_cols))]
    )

    if manual_sel and len(manual_sel) >= 1:
        with st.spinner("Running 5-fold cross-validation…"):
            rf_all = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            rf_sel = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

            cv_all = cross_val_score(rf_all, X_raw, y_raw, cv=5, scoring="r2").mean()
            X_sel = X_raw[manual_sel]
            cv_sel = cross_val_score(rf_sel, X_sel, y_raw, cv=5, scoring="r2").mean()

        c1, c2, c3 = st.columns(3)
        c1.metric("All Features R²", f"{cv_all:.4f}", f"{len(predictor_cols)} features")
        c2.metric("Selected Subset R²", f"{cv_sel:.4f}", f"{len(manual_sel)} features")
        c3.metric("R² Difference", f"{cv_sel - cv_all:+.4f}",
                  delta_color="normal" if cv_sel >= cv_all * 0.95 else "inverse")

        fig_val = go.Figure(go.Bar(
            x=["All Features", "Selected Subset"],
            y=[cv_all, cv_sel],
            marker_color=["#38bdf8", "#4ade80"],
            text=[f"{cv_all:.4f}", f"{cv_sel:.4f}"],
            textposition="outside"
        ))
        fig_val.update_layout(
            title="5-Fold CV R² Comparison",
            template="plotly_dark",
            yaxis=dict(range=[0, 1.1], title="R² Score"),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_val, use_container_width=True)
    else:
        st.info("Select at least one feature above to run validation.")

# ── Save Selection ────────────────────────────────────────────────────────────
st.markdown("---")
col_save1, col_save2 = st.columns([3, 1])
with col_save1:
    final_selection = st.multiselect(
        "💾 Final Feature Subset to Save to Session",
        predictor_cols,
        default=combined_sorted.index[:min(3, len(combined_sorted))].tolist()
    )
with col_save2:
    st.markdown("")
    st.markdown("")
    if st.button("Save Feature Subset", use_container_width=True):
        st.session_state.selected_features = final_selection
        st.success(f"Saved {len(final_selection)} features to session: {final_selection}")

render_footer()
