"""Shared training / evaluation helpers for Phase 1–2 experiments."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from metrics import regression_metrics
from model import RevenueModel
from utils import (
    FEATURE_COLS,
    TARGET_COL,
    dataframe_to_xy,
    predict_from_residual,
)


def model_nbytes(model: Optional[RevenueModel] = None) -> int:
    """Float32 parameter payload size in bytes."""
    m = model or RevenueModel()
    return int(sum(p.numel() * 4 for p in m.parameters()))


def train_residual_mlp(
    X: torch.Tensor,
    y: torch.Tensor,
    X_val: Optional[torch.Tensor] = None,
    y_val: Optional[torch.Tensor] = None,
    epochs: int = 400,
    patience: int = 30,
    lr: float = 1e-3,
) -> RevenueModel:
    model = RevenueModel()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad = 0
    use_val = X_val is not None and y_val is not None and len(X_val) > 0

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()

        if not use_val:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            continue

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()
        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_model(
    model: RevenueModel,
    X_test: torch.Tensor,
    y_test_residual: torch.Tensor,
    lag12_test: np.ndarray,
) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        resid = model(X_test)
        pred = predict_from_residual(resid, lag12_test)
        y_true = predict_from_residual(y_test_residual, lag12_test)
    return regression_metrics(y_true, pred)


def prepare_xy(
    train_df,
    test_df,
    x_scaler: StandardScaler,
    val_df=None,
) -> Tuple:
    X_train, y_train = dataframe_to_xy(train_df, x_scaler)
    X_test, y_test = dataframe_to_xy(test_df, x_scaler)
    lag12_test = test_df["gst_lag_12"].values.astype(float)
    if val_df is not None and len(val_df) > 0:
        X_val, y_val = dataframe_to_xy(val_df, x_scaler)
        return X_train, y_train, X_val, y_val, X_test, y_test, lag12_test
    return X_train, y_train, None, None, X_test, y_test, lag12_test


def assert_feature_schema():
    assert FEATURE_COLS and TARGET_COL
