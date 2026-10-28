"""Forecast error metrics in original units."""

import numpy as np


def regression_metrics(y_true, y_pred):
    """Return RMSE, MAE and MAPE (percent, over nonzero targets)."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    error = predicted - actual
    nonzero = actual != 0
    mape = (
        float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100.0)
        if nonzero.any()
        else float("nan")
    )
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "mape": mape,
    }
