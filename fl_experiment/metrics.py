"""Forecast and classification evaluation metrics."""

from typing import Dict, Union

import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy(y: ArrayLike) -> np.ndarray:
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    return np.asarray(y, dtype=float).reshape(-1)


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    yt, yp = _to_numpy(y_true), _to_numpy(y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    yt, yp = _to_numpy(y_true), _to_numpy(y_pred)
    return float(np.mean(np.abs(yt - yp)))


def mape(y_true: ArrayLike, y_pred: ArrayLike, eps: float = 1e-8) -> float:
    yt, yp = _to_numpy(y_true), _to_numpy(y_pred)
    denom = np.maximum(np.abs(yt), eps)
    return float(np.mean(np.abs((yt - yp) / denom)) * 100.0)


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> Dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }


def classification_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    scores: ArrayLike = None,
) -> Dict[str, float]:
    yt = _to_numpy(y_true).astype(int)
    yp = _to_numpy(y_pred).astype(int)
    out = {
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall": float(recall_score(yt, yp, zero_division=0)),
        "f1": float(f1_score(yt, yp, zero_division=0)),
        "anomaly_rate_pred": float(yp.mean()) if len(yp) else 0.0,
        "anomaly_rate_true": float(yt.mean()) if len(yt) else 0.0,
    }
    if scores is not None and len(np.unique(yt)) > 1:
        out["roc_auc"] = float(roc_auc_score(yt, _to_numpy(scores)))
    else:
        out["roc_auc"] = float("nan")
    return out
