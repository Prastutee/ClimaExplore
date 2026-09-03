"""
Phase 2: Analyze — Resampling & Modification Page
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from utils.ui import render_header, render_footer
from utils.data_loader import resample_dataset, apply_transform, df_to_csv_bytes

st.set_page_config(page_title="Resampling & Modification — ClimaXplore", page_icon="🔄", layout="wide")
render_header(title="Resampling & Dataset Modification",
              subtitle="Frequency Resampling · Train-Test Split · Log/Z-Score Transforms · CSV Export")

st.title("🔄 Resampling & Dataset Modification")

df = st.session_state.df_clean
datetime_col = st.session_state.get("datetime_col", "Datetime")
target_col   = st.session_state.get("target_col", "PRECTOTCORR")

if df is None or df.empty:
    st.error("No dataset available. Please upload data on the Data Upload page first.")
    st.stop()

numeric_cols = [c for c in df.columns if c != datetime_col and pd.api.types.is_numeric_dtype(df[c])]

tab_resamp, tab_split, tab_transform, tab_export = st.tabs([
    "📅 Resampling", "✂️ Train-Test Split", "🔧 Transforms", "💾 Export"
])

# ── 1. Resampling ─────────────────────────────────────────────────────────────
with tab_resamp:
    st.subheader("Frequency Resampling Options")
    col1, col2 = st.columns(2)
    with col1:
        resample_freq = st.selectbox("Resampling Frequency", ["Daily", "Monthly", "Annual"])
    with col2:
        agg_func = st.selectbox("Aggregation Function", ["mean", "sum", "max", "min"])

    col_before, col_after = st.columns(2)
    with col_before:
        st.metric("Current Rows", f"{len(df):,}")
    with col_after:
        st.metric("Current Frequency", "as-is")

    if st.button("⚡ Apply Frequency Resampling", use_container_width=True):
        df_resampled = resample_dataset(df, datetime_col, freq=resample_freq, agg_func=agg_func)
        st.session_state.df_clean = df_resampled
        n_train = int(len(df_resampled) * (st.session_state.get("split_ratio", 80) / 100))
        st.session_state.df_train = df_resampled.iloc[:n_train]
        st.session_state.df_test  = df_resampled.iloc[n_train:]
        st.success(f"✅ Resampled to **{resample_freq}** ({agg_func}) — {len(df_resampled):,} rows remaining.")
        df = df_resampled

    # Preview before/after
    st.markdown("#### Dataset Preview")
    st.dataframe(st.session_state.df_clean.head(30), use_container_width=True)

    # Resampled line chart
    if datetime_col in st.session_state.df_clean.columns and target_col in st.session_state.df_clean.columns:
        df_plot = st.session_state.df_clean[[datetime_col, target_col]].dropna()
        fig_rs = px.line(df_plot, x=datetime_col, y=target_col,
                         title=f"{target_col} — Resampled View",
                         template="plotly_dark",
                         color_discrete_sequence=["#38bdf8"])
        fig_rs.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_rs, use_container_width=True)

# ── 2. Train-Test Split ───────────────────────────────────────────────────────
with tab_split:
    st.subheader("Train-Test Split Configuration")
    split_ratio = st.slider("Train Set Proportion (%)", 50, 90, st.session_state.get("split_ratio", 80), step=5)

    df_cur = st.session_state.df_clean
    split_idx = int(len(df_cur) * (split_ratio / 100.0))
    df_train  = df_cur.iloc[:split_idx]
    df_test   = df_cur.iloc[split_idx:]

    st.session_state.df_train   = df_train
    st.session_state.df_test    = df_test
    st.session_state.split_ratio = split_ratio

    # Metric row
    c1, c2, c3 = st.columns(3)
    c1.metric("Training Samples", f"{len(df_train):,}", f"{split_ratio}% of data")
    c2.metric("Testing Samples",  f"{len(df_test):,}",  f"{100 - split_ratio}% of data")
    c3.metric("Total Samples",    f"{len(df_cur):,}")

    # Visual timeline chart
    if datetime_col in df_cur.columns and pd.api.types.is_datetime64_any_dtype(df_cur[datetime_col]) \
       and target_col in df_cur.columns:
        fig_split = go.Figure()
        fig_split.add_trace(go.Scatter(
            x=df_train[datetime_col], y=df_train[target_col],
            name="Training Data",
            line=dict(color="#38bdf8", width=1.5),
            fill="tozeroy", fillcolor="rgba(56,189,248,0.08)"
        ))
        fig_split.add_trace(go.Scatter(
            x=df_test[datetime_col], y=df_test[target_col],
            name="Test Data",
            line=dict(color="#fbbf24", width=1.5),
            fill="tozeroy", fillcolor="rgba(251,191,36,0.08)"
        ))
        if len(df_train) > 0 and len(df_test) > 0:
            split_date = df_test[datetime_col].iloc[0]
            fig_split.add_vline(x=split_date, line_dash="dash", line_color="#f87171",
                                annotation_text=f"Split: {split_date.date()}")
        fig_split.update_layout(
            title=f"Train-Test Split Timeline — {target_col}",
            template="plotly_dark",
            xaxis_title="Date",
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_split, use_container_width=True)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.caption("**Training Set — First 10 rows**")
        st.dataframe(df_train.head(10), use_container_width=True)
    with col_t2:
        st.caption("**Test Set — Last 10 rows**")
        st.dataframe(df_test.tail(10), use_container_width=True)

# ── 3. Transforms ─────────────────────────────────────────────────────────────
with tab_transform:
    st.subheader("Column Transforms")
    st.info("Transforms are applied to the active session dataset. You can re-apply after resampling.")

    col_tr1, col_tr2, col_tr3 = st.columns(3)
    with col_tr1:
        transform_col = st.selectbox("Column to Transform", numeric_cols)
    with col_tr2:
        transform_method = st.selectbox("Transform Method", ["Log (log1p)", "Sqrt", "Z-Score", "MinMax", "Differencing"])
    with col_tr3:
        new_col_name = st.text_input("New Column Name", value=f"{transform_col}_{transform_method.split()[0].lower()}")

    method_map = {
        "Log (log1p)": "Log", "Sqrt": "Sqrt",
        "Z-Score": "Z-Score", "MinMax": "MinMax", "Differencing": "Differencing"
    }

    if st.button("🔧 Apply Transform & Add Column", use_container_width=True):
        df_upd = st.session_state.df_clean.copy()
        df_upd[new_col_name] = apply_transform(df_upd[transform_col], method_map[transform_method]).values
        st.session_state.df_clean = df_upd
        st.success(f"✅ Added column **{new_col_name}** using {transform_method}.")

        # Preview
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(y=df_upd[transform_col], name="Original", line=dict(color="#64748b")))
        fig_cmp.add_trace(go.Scatter(y=df_upd[new_col_name], name=f"Transformed ({transform_method})", line=dict(color="#38bdf8")))
        fig_cmp.update_layout(
            title="Original vs. Transformed",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

# ── 4. Export ─────────────────────────────────────────────────────────────────
with tab_export:
    st.subheader("Export Datasets")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("##### 📦 Full Cleaned Dataset")
        st.download_button(
            "📥 Download Full Dataset",
            data=df_to_csv_bytes(st.session_state.df_clean),
            file_name="climaxplore_full_clean.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.caption(f"{len(st.session_state.df_clean):,} rows × {len(st.session_state.df_clean.columns)} columns")

    with c2:
        st.markdown("##### 🎓 Training Set")
        st.download_button(
            "📥 Download Training Set",
            data=df_to_csv_bytes(st.session_state.df_train),
            file_name="climaxplore_train.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.caption(f"{len(st.session_state.df_train):,} rows — {st.session_state.get('split_ratio', 80)}%")

    with c3:
        st.markdown("##### 🧪 Test Set")
        st.download_button(
            "📥 Download Test Set",
            data=df_to_csv_bytes(st.session_state.df_test),
            file_name="climaxplore_test.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.caption(f"{len(st.session_state.df_test):,} rows — {100 - st.session_state.get('split_ratio', 80)}%")

    st.markdown("#### Preview — Current Dataset")
    st.dataframe(st.session_state.df_clean.head(25), use_container_width=True)

render_footer()
