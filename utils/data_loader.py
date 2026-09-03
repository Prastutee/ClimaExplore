"""
Data Loader and Imputation Module for ClimaXplore.

Handles file loading, datetime parsing, missing value imputation (Mean, Median, Mode, KNN),
dataset resampling, and data transforms.
"""

import os
import io
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


def load_default_sample():
    """Load default NASA POWER climate sample dataset or generate synthetic fallback."""
    sample_path = os.path.join(os.getcwd(), "sample_data", "nasa_power_climate_sample.csv")
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
        if "Datetime" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Datetime"])
        return df
    else:
        # Generate synthetic NASA POWER-style daily dataset (3 years)
        dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")
        n = len(dates)
        np.random.seed(42)
        t = np.linspace(0, 6 * np.pi, n)

        t2m = 22 + 10 * np.sin(t) + np.random.normal(0, 1.5, n)
        prectot = np.maximum(0, 40 * np.sin(np.linspace(0, 3 * np.pi, n)) ** 4 + np.random.normal(0, 8, n))
        ws2m = np.abs(3.2 + 1.5 * np.cos(t * 0.5) + np.random.normal(0, 0.8, n))
        rh2m = np.clip(65 + 20 * np.cos(t) + np.random.normal(0, 5, n), 20, 100)
        sw_dw = np.clip(5.5 + 3.0 * np.sin(t) + np.random.normal(0, 0.5, n), 0.5, 10.0)
        t2m_min = t2m - np.abs(np.random.normal(4, 1.5, n))
        t2m_max = t2m + np.abs(np.random.normal(4, 1.5, n))

        df = pd.DataFrame({
            "Datetime": dates,
            "T2M": np.round(t2m, 3),
            "T2M_MIN": np.round(t2m_min, 3),
            "T2M_MAX": np.round(t2m_max, 3),
            "PRECTOTCORR": np.round(prectot, 3),
            "WS2M": np.round(ws2m, 3),
            "RH2M": np.round(rh2m, 2),
            "ALLSKY_SFC_SW_DW": np.round(sw_dw, 3),
        })
        return df


def impute_missing_values(df, numeric_cols, method="Mean"):
    """
    Impute missing values in numeric columns.

    Methods: Mean, Median, Mode, KNN
    """
    df_imputed = df.copy()

    if method == "Mean":
        for col in numeric_cols:
            if df_imputed[col].isnull().any():
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())
    elif method == "Median":
        for col in numeric_cols:
            if df_imputed[col].isnull().any():
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
    elif method == "Mode":
        for col in numeric_cols:
            if df_imputed[col].isnull().any():
                mode_val = df_imputed[col].mode()[0] if not df_imputed[col].mode().empty else 0
                df_imputed[col] = df_imputed[col].fillna(mode_val)
    elif method == "KNN":
        imputer = KNNImputer(n_neighbors=5)
        df_imputed[numeric_cols] = imputer.fit_transform(df_imputed[numeric_cols])

    return df_imputed


def resample_dataset(df, datetime_col, freq="Daily", agg_func="mean"):
    """Resample dataset by frequency ('Daily', 'Monthly', 'Annual')."""
    df_resampled = df.copy()
    if datetime_col not in df_resampled.columns:
        return df_resampled

    df_resampled = df_resampled.sort_values(datetime_col)
    df_resampled.set_index(datetime_col, inplace=True)

    rule_map = {"Daily": "D", "Monthly": "MS", "Annual": "YS"}
    rule = rule_map.get(freq, "D")

    numeric_cols = df_resampled.select_dtypes(include=[np.number]).columns
    agg_map = {"mean": "mean", "sum": "sum", "max": "max", "min": "min"}
    fn = agg_map.get(agg_func, "mean")
    resampled_df = df_resampled[numeric_cols].resample(rule).agg(fn)

    return resampled_df.reset_index()


def apply_transform(series: pd.Series, method: str) -> pd.Series:
    """
    Apply a mathematical transform to a numeric series.

    Methods: 'Log', 'Sqrt', 'Z-Score', 'MinMax', 'Differencing'
    """
    s = series.copy()
    if method == "Log":
        s = np.log1p(s.clip(lower=0))
    elif method == "Sqrt":
        s = np.sqrt(s.clip(lower=0))
    elif method == "Z-Score":
        s = (s - s.mean()) / (s.std() + 1e-8)
    elif method == "MinMax":
        s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    elif method == "Differencing":
        s = s.diff().fillna(0)
    return s


def get_missing_summary(df):
    """Return a summary DataFrame of missing values per column."""
    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    dtypes = df.dtypes.astype(str)
    summary = pd.DataFrame({
        "Column": null_counts.index,
        "Missing Count": null_counts.values,
        "Missing %": null_pct.values,
        "Dtype": dtypes.values,
    }).sort_values("Missing Count", ascending=False).reset_index(drop=True)
    return summary


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to UTF-8 encoded CSV bytes for download."""
    return df.to_csv(index=False).encode("utf-8")
