"""Data loading, forecast-frame construction, and non-IID helpers."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parent / "data" / "state_budget_gst.csv"

FEATURE_COLS = [
    "gst_lag_1",
    "gst_lag_12",
    "month",
    "fiscal_deficit_pct",
    "urban_ratio",
    "gdp_growth_pct",
]
TARGET_COL = "target"  # next-month gst_revenue

TRAIN_END_YEAR = 2021
VAL_YEAR = 2022
TEST_START_YEAR = 2023

CLIENT_DATA = {}


def load_dataframe() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Run data-generator/generate_state_budget_gst.py first."
        )
    return pd.read_csv(DATA_PATH)


def build_forecast_frame(df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Build a 1-month-ahead forecasting panel per state.

    At origin month t, features include current GST and GST from 11 months
    earlier; target is GST at t+1.
    """
    if df is None:
        df = load_dataframe()

    frames = []
    for _, group in df.groupby("state_id", sort=True):
        g = group.sort_values(["year", "month"]).copy()
        g["gst_lag_1"] = g["gst_revenue"]
        g["gst_lag_12"] = g["gst_revenue"].shift(11)
        g["target"] = g["gst_revenue"].shift(-1)
        frames.append(g)

    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=["gst_lag_12", "target"]).reset_index(drop=True)


def temporal_masks(df: pd.DataFrame):
    train = df["year"] <= TRAIN_END_YEAR
    val = df["year"] == VAL_YEAR
    test = df["year"] >= TEST_START_YEAR
    return train, val, test


def residual_target(df: pd.DataFrame) -> np.ndarray:
    """log1p(target) - log1p(seasonal naive) — model learns corrections only."""
    y = np.log1p(df[TARGET_COL].values.astype(float))
    base = np.log1p(df["gst_lag_12"].values.astype(float))
    return y - base


def predict_from_residual(residual_log: torch.Tensor, gst_lag_12: np.ndarray) -> np.ndarray:
    base = np.log1p(np.asarray(gst_lag_12, dtype=float).reshape(-1))
    resid = residual_log.detach().cpu().numpy().reshape(-1)
    return np.expm1(base + resid)


def fit_feature_scaler(train_df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLS].values.astype(float))
    return scaler


def dataframe_to_xy(
    df: pd.DataFrame,
    x_scaler: StandardScaler,
) -> Tuple[torch.Tensor, torch.Tensor]:
    X = x_scaler.transform(df[FEATURE_COLS].values.astype(float))
    y = residual_target(df)
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).view(-1, 1),
    )


def load_state_data(state_id: int):
    """
    Returns (X_train, y_train, X_test, y_test, lag12_test).
    y is residual in log-space vs seasonal naive; lag12_test used to decode preds.
    """
    df = build_forecast_frame()
    df = df[df["state_id"] == state_id]
    if df.empty:
        raise ValueError(f"No forecast rows for state_id={state_id}")

    train_mask, _, test_mask = temporal_masks(df)
    train_df = df.loc[train_mask]
    test_df = df.loc[test_mask]
    if train_df.empty or test_df.empty:
        raise ValueError(
            f"state_id={state_id} missing train or test rows after temporal split "
            f"(train_end={TRAIN_END_YEAR}, test_start={TEST_START_YEAR})"
        )

    x_scaler = fit_feature_scaler(train_df)
    X_train, y_train = dataframe_to_xy(train_df, x_scaler)
    X_test, y_test = dataframe_to_xy(test_df, x_scaler)
    lag12_test = test_df["gst_lag_12"].values.astype(float)
    return X_train, y_train, X_test, y_test, lag12_test


def naive_seasonal_predict(df: pd.DataFrame) -> np.ndarray:
    return df["gst_lag_12"].values.astype(float)


def set_client_data(split_data: dict):
    global CLIENT_DATA
    CLIENT_DATA = split_data


def load_client_data(client_id: int):
    if client_id not in CLIENT_DATA:
        raise KeyError(f"No partition for client_id={client_id}. Call set_client_data first.")
    client_df = CLIENT_DATA[client_id]
    x_scaler = fit_feature_scaler(client_df)
    return dataframe_to_xy(client_df, x_scaler)


def iid_split(df: pd.DataFrame, num_clients: int) -> dict:
    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    splits = np.array_split(shuffled, num_clients)
    return {i: split.reset_index(drop=True) for i, split in enumerate(splits)}


def quantity_skew_split(df: pd.DataFrame, ratios: list) -> dict:
    ratios = np.asarray(ratios, dtype=float)
    probs = ratios / ratios.sum()
    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    assignments = np.random.default_rng(42).choice(len(probs), size=len(shuffled), p=probs)
    return {
        i: shuffled[assignments == i].reset_index(drop=True) for i in range(len(probs))
    }


def label_skew_split(df: pd.DataFrame, num_clients: int) -> dict:
    df_sorted = df.sort_values(TARGET_COL).reset_index(drop=True)
    splits = np.array_split(df_sorted, num_clients)
    return {i: split.reset_index(drop=True) for i, split in enumerate(splits)}


def feature_skew_split(df: pd.DataFrame) -> dict:
    threshold = df["urban_ratio"].median()
    return {
        0: df[df["urban_ratio"] > threshold].reset_index(drop=True),
        1: df[df["urban_ratio"] <= threshold].reset_index(drop=True),
    }
