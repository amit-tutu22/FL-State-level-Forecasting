"""Generic forecast-frame builder driven by TaskSpec."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from task_spec import TaskSpec


def load_raw(spec: TaskSpec) -> pd.DataFrame:
    path = spec.resolve_data_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Sector data missing: {path}. Generate it for sector '{spec.sector_id}'."
        )
    return pd.read_csv(path)


def build_forecast_frame(spec: TaskSpec, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if df is None:
        df = load_raw(spec)

    required = [spec.client_id_col, spec.value_col, *spec.time_cols, *spec.covariate_cols]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Sector '{spec.sector_id}' missing columns: {missing}")

    frames = []
    sort_cols = list(spec.time_cols)
    for _, group in df.groupby(spec.client_id_col, sort=True):
        g = group.sort_values(sort_cols).copy()
        for lag in spec.lag_months:
            # lag k relative to target at t+horizon ≈ shift(lag - horizon) from origin row
            # origin row t predicts t+horizon; value at t+horizon-lag is shift(lag - horizon)
            shift = lag - spec.horizon
            g[f"{spec.value_col}_lag_{lag}"] = g[spec.value_col].shift(shift)
        g["target"] = g[spec.value_col].shift(-spec.horizon)
        frames.append(g)

    out = pd.concat(frames, ignore_index=True)
    drop_cols = [f"{spec.value_col}_lag_{lag}" for lag in spec.lag_months] + ["target"]
    return out.dropna(subset=drop_cols).reset_index(drop=True)


def temporal_masks(spec: TaskSpec, df: pd.DataFrame):
    year_col = "year" if "year" in df.columns else spec.time_cols[0]
    train = df[year_col] <= spec.train_end_year
    val = df[year_col] == spec.val_year
    test = df[year_col] >= spec.test_start_year
    return train, val, test


def residual_target(spec: TaskSpec, df: pd.DataFrame) -> np.ndarray:
    y = np.log1p(df["target"].values.astype(float))
    if spec.residual_on_seasonal_lag and spec.seasonal_lag_col:
        base = np.log1p(df[spec.seasonal_lag_col].values.astype(float))
        return y - base
    return y


def predict_from_model_output(
    spec: TaskSpec,
    residual_or_log: torch.Tensor,
    df_slice: pd.DataFrame,
) -> np.ndarray:
    arr = residual_or_log.detach().cpu().numpy().reshape(-1)
    if spec.residual_on_seasonal_lag and spec.seasonal_lag_col:
        base = np.log1p(df_slice[spec.seasonal_lag_col].values.astype(float))
        return np.expm1(base + arr)
    return np.expm1(arr)


def naive_seasonal_predict(spec: TaskSpec, df: pd.DataFrame) -> np.ndarray:
    if not spec.seasonal_lag_col:
        raise ValueError("Seasonal naive requires lag 12 in TaskSpec.lag_months")
    return df[spec.seasonal_lag_col].values.astype(float)


def fit_feature_scaler(spec: TaskSpec, train_df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[spec.feature_cols].values.astype(float))
    return scaler


def dataframe_to_xy(
    spec: TaskSpec,
    df: pd.DataFrame,
    x_scaler: StandardScaler,
) -> Tuple[torch.Tensor, torch.Tensor]:
    X = x_scaler.transform(df[spec.feature_cols].values.astype(float))
    y = residual_target(spec, df)
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).view(-1, 1),
    )


def client_partitions(spec: TaskSpec, train_df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """Natural federation: one client per entity id."""
    parts = {}
    for i, cid in enumerate(sorted(train_df[spec.client_id_col].unique())):
        parts[i] = train_df[train_df[spec.client_id_col] == cid].reset_index(drop=True)
    return parts
