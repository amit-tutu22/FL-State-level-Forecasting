"""Data loading and non-IID partition helpers for the FL experiment."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parent / "data" / "state_budget_gst.csv"

# Avoid sgst_share/cgst_share: they are deterministic fractions of gst_revenue.
FEATURE_COLS = [
    "year",
    "month",
    "fiscal_deficit_pct",
    "urban_ratio",
    "gdp_growth_pct",
]
TARGET_COL = "gst_revenue"

CLIENT_DATA = {}


def load_dataframe() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Run data-generator/generate_state_budget_gst.py first."
        )
    return pd.read_csv(DATA_PATH)


def _to_tensors(features: np.ndarray, target: np.ndarray, fit_scaler: bool = True, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        features = scaler.fit_transform(features)
    else:
        features = scaler.transform(features)
    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(target, dtype=torch.float32).view(-1, 1),
        scaler,
    )


def load_state_data(state_id: int):
    """Load one state's rows as tensors (Phase 0 FL client path)."""
    df = load_dataframe()
    df = df[df["state_id"] == state_id]
    if df.empty:
        raise ValueError(f"No rows for state_id={state_id}")

    X, y, _ = _to_tensors(
        df[FEATURE_COLS].values,
        df[TARGET_COL].values,
    )
    return X, y


def set_client_data(split_data: dict):
    global CLIENT_DATA
    CLIENT_DATA = split_data


def load_client_data(client_id: int):
    """Load a pre-partitioned client dataframe as tensors."""
    if client_id not in CLIENT_DATA:
        raise KeyError(f"No partition for client_id={client_id}. Call set_client_data first.")
    client_df = CLIENT_DATA[client_id]
    X, y, _ = _to_tensors(
        client_df[FEATURE_COLS].values,
        client_df[TARGET_COL].values,
    )
    return X, y


def iid_split(df: pd.DataFrame, num_clients: int) -> dict:
    """Disjoint random partition across clients."""
    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    splits = np.array_split(shuffled, num_clients)
    return {i: splits[i].reset_index(drop=True) for i in range(num_clients)}


def quantity_skew_split(df: pd.DataFrame, ratios: list) -> dict:
    """
    Disjoint quantity skew: assign each row to one client with probability
    proportional to ratios, then optionally downsample is unnecessary because
    assignment already controls relative sizes.
    """
    ratios = np.asarray(ratios, dtype=float)
    probs = ratios / ratios.sum()
    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    assignments = np.random.default_rng(42).choice(len(probs), size=len(shuffled), p=probs)
    clients = {}
    for i in range(len(probs)):
        clients[i] = shuffled[assignments == i].reset_index(drop=True)
    return clients


def label_skew_split(df: pd.DataFrame, num_clients: int) -> dict:
    """Disjoint label skew: sort by target and split into contiguous bands."""
    df_sorted = df.sort_values(TARGET_COL).reset_index(drop=True)
    splits = np.array_split(df_sorted, num_clients)
    return {i: splits[i].reset_index(drop=True) for i in range(num_clients)}


def feature_skew_split(df: pd.DataFrame) -> dict:
    """Disjoint feature skew: urban vs non-urban (median split for balance)."""
    threshold = df["urban_ratio"].median()
    return {
        0: df[df["urban_ratio"] > threshold].reset_index(drop=True),
        1: df[df["urban_ratio"] <= threshold].reset_index(drop=True),
    }
