"""
Phase 1: Visualize — Data Upload & Cleaning Page
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils.ui import render_header, render_footer, render_info
from utils.data_loader import impute_missing_values, get_missing_summary

st.set_page_config(page_title="Data Upload & Cleaning — ClimaXplore", page_icon="📁", layout="wide")
render_header(title="Data Upload & Cleaning", subtitle="Upload CSV · Map Columns · Impute Missing Values")

st.title("📁 Data Upload & Cleaning")

# ── 1. File Upload ─────────────────────────────────────────────────────────────
st.markdown('<div class="cx-phase-header"><span class="cx-phase-label">Step 1</span><span class="cx-phase-title"> — Load Dataset</span></div>', unsafe_allow_html=True)

col_upload, col_info = st.columns([2, 1])
with col_upload:
    uploaded_file = st.file_uploader(
        "Upload a Climate CSV file (NASA POWER or compatible format)",
        type=["csv"],
        help="The file must contain a date/datetime column and one or more numeric climate variables."
    )
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.session_state.df_raw = df_upload
            st.success(f"✅ Uploaded **{uploaded_file.name}** — {len(df_upload):,} rows × {len(df_upload.columns)} columns")
        except Exception as e:
            st.error(f"Failed to read file: {e}")

with col_info:
    st.markdown(
        """<div class="cx-card">
            <div class="cx-metric-label">💡 Expected Format</div>
            <ul style="color:#94a3b8;font-size:13px;margin-top:8px;padding-left:16px;">
                <li>One Datetime column (ISO format preferred)</li>
                <li>One or more numeric climate columns</li>
                <li>NASA POWER params: T2M, PRECTOTCORR, WS2M, RH2M…</li>
                <li>CSV with a header row</li>
            </ul>
        </div>""",
        unsafe_allow_html=True
    )

df = st.session_state.df_raw.copy()

# ── 2. Column Mapping ─────────────────────────────────────────────────────────
st.markdown('<div class="cx-phase-header"><span class="cx-phase-label">Step 2</span><span class="cx-phase-title"> — Column Mapping & Variable Selection</span></div>', unsafe_allow_html=True)

all_cols = list(df.columns)
numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]
datetime_candidates = [c for c in all_cols if "date" in c.lower() or "time" in c.lower() or "dt" in c.lower()]

col1, col2, col3 = st.columns(3)
with col1:
    default_dt = datetime_candidates[0] if datetime_candidates else all_cols[0]
    datetime_col = st.selectbox("📅 Datetime Column", all_cols,
                                 index=all_cols.index(default_dt) if default_dt in all_cols else 0)
    st.session_state.datetime_col = datetime_col
    try:
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        st.success(f"Parsed '{datetime_col}' as datetime ✓")
    except Exception as e:
        st.warning(f"Could not parse '{datetime_col}': {e}")

with col2:
    numeric_excl = [c for c in all_cols if c != datetime_col and pd.api.types.is_numeric_dtype(df[c])]
    default_target = (
        "T2M" if "T2M" in numeric_excl else
        "PRECTOTCORR" if "PRECTOTCORR" in numeric_excl else
        numeric_excl[0] if numeric_excl else all_cols[0]
    )
    target_col = st.selectbox("🎯 Primary Target Variable", numeric_excl if numeric_excl else all_cols,
                               index=numeric_excl.index(default_target) if default_target in numeric_excl else 0)
    st.session_state.target_col = target_col

with col3:
    selected_vars = st.multiselect(
        "📊 Weather Variables to Keep",
        numeric_excl,
        default=numeric_excl,
        help="Only selected variables will be retained in the cleaned dataset."
    )

# Dataset overview
st.markdown("#### Dataset Overview")
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Rows", f"{len(df):,}")
col_b.metric("Columns", str(len(df.columns)))
col_c.metric("Numeric Cols", str(len(numeric_excl)))
date_range = "N/A"
if datetime_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
    date_range = f"{df[datetime_col].min().date()} → {df[datetime_col].max().date()}"
col_d.metric("Date Range", date_range)

# ── 3. Missing Value Analysis ─────────────────────────────────────────────────
st.markdown('<div class="cx-phase-header"><span class="cx-phase-label">Step 3</span><span class="cx-phase-title"> — Missing Value Analysis</span></div>', unsafe_allow_html=True)

miss_summary = get_missing_summary(df)
total_missing = int(miss_summary["Missing Count"].sum())

if total_missing > 0:
    st.warning(f"⚠️ Dataset contains **{total_missing:,}** missing values across {(miss_summary['Missing Count'] > 0).sum()} columns.")
    col_table, col_bar = st.columns([1, 2])
    with col_table:
        st.dataframe(miss_summary, use_container_width=True, hide_index=True)
    with col_bar:
        fig_null = px.bar(
            miss_summary[miss_summary["Missing Count"] > 0],
            x="Column", y="Missing %",
            title="Missing Value Rate by Column (%)",
            template="plotly_dark",
            color="Missing %",
            color_continuous_scale="Reds",
            text_auto=".1f"
        )
        fig_null.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=20)
        )
        st.plotly_chart(fig_null, use_container_width=True)
else:
    render_info("No missing values detected in the dataset — it is complete!", "success")

# ── 4. Imputation ─────────────────────────────────────────────────────────────
st.markdown('<div class="cx-phase-header"><span class="cx-phase-label">Step 4</span><span class="cx-phase-title"> — Impute & Apply Cleaning</span></div>', unsafe_allow_html=True)

col_imp1, col_imp2 = st.columns([2, 1])
with col_imp1:
    impute_method = st.selectbox(
        "🔧 Missing Value Imputation Method",
        ["None / Keep Existing", "Mean", "Median", "Mode", "KNN Imputation"],
        help="KNN uses the 5 nearest neighbours based on numeric values to fill gaps."
    )
with col_imp2:
    st.markdown("")
    st.markdown("")
    apply_btn = st.button("✅ Apply Cleaning & Save to Session", use_container_width=True)

if apply_btn:
    keep_cols = [datetime_col] + (selected_vars if selected_vars else numeric_excl)
    existing_keep = [c for c in keep_cols if c in df.columns]
    df_clean = df[existing_keep].copy()

    if impute_method != "None / Keep Existing" and numeric_excl:
        num_to_impute = [c for c in numeric_excl if c in df_clean.columns]
        method_name = impute_method.replace(" Imputation", "")
        df_clean = impute_missing_values(df_clean, num_to_impute, method=method_name)
        st.success(f"✅ Imputed using **{impute_method}** — {df_clean.isnull().sum().sum()} missing values remain.")

    st.session_state.df_clean = df_clean
    n_train = int(len(df_clean) * (st.session_state.get("split_ratio", 80) / 100))
    st.session_state.df_train = df_clean.iloc[:n_train]
    st.session_state.df_test  = df_clean.iloc[n_train:]
    render_info(f"Session state updated — {len(df_clean):,} rows × {len(df_clean.columns)} columns saved.", "success")

# ── 5. Preview ────────────────────────────────────────────────────────────────
st.markdown('<div class="cx-phase-header"><span class="cx-phase-label">Step 5</span><span class="cx-phase-title"> — Dataset Preview</span></div>', unsafe_allow_html=True)

tab_head, tab_tail, tab_stats = st.tabs(["🔼 Head (50 rows)", "🔽 Tail (20 rows)", "📈 Describe"])
with tab_head:
    st.dataframe(st.session_state.df_clean.head(50), use_container_width=True)
with tab_tail:
    st.dataframe(st.session_state.df_clean.tail(20), use_container_width=True)
with tab_stats:
    st.dataframe(
        st.session_state.df_clean.describe().T.style.background_gradient(cmap="Blues"),
        use_container_width=True
    )

render_footer()
