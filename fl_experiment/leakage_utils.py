"""Welfare leakage / anomaly-detection data helpers."""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parent / "data" / "welfare_leakage.csv"

FEATURE_COLS = [
    "claim_amount",
    "beneficiary_count",
    "avg_days_worked",
    "prior_month_claim",
    "urban_ratio",
    "month",
    "claim_to_prior_ratio",
    "claim_per_beneficiary",
]
LABEL_COL = "is_anomaly"
SEED = 42


def load_welfare_dataframe() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Run data-generator/generate_welfare_leakage.py first."
        )
    df = pd.read_csv(DATA_PATH)
    df = df.copy()
    df["claim_to_prior_ratio"] = df["claim_amount"] / np.maximum(df["prior_month_claim"], 1.0)
    df["claim_per_beneficiary"] = df["claim_amount"] / np.maximum(df["beneficiary_count"], 1.0)
    return df


def temporal_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train ≤2021, val=2022, test≥2023 (by claim year)."""
    train = df[df["year"] <= 2021].reset_index(drop=True)
    val = df[df["year"] == 2022].reset_index(drop=True)
    test = df[df["year"] >= 2023].reset_index(drop=True)
    return train, val, test


def fit_scaler(train_df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLS].values.astype(float))
    return scaler


def transform_X(df: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(df[FEATURE_COLS].values.astype(float))


def labels(df: pd.DataFrame) -> np.ndarray:
    return df[LABEL_COL].values.astype(int)


def normal_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df[LABEL_COL] == 0].reset_index(drop=True)


def state_partitions(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    return {
        int(sid): g.reset_index(drop=True)
        for sid, g in df.groupby("state_id")
    }


def to_tensor(X: np.ndarray) -> torch.Tensor:
    return torch.tensor(X, dtype=torch.float32)
