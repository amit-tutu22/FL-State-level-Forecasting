"""Forecast error metrics in original units."""

import numpy as np


def _columns(y_true, y_pred):
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted lengths differ")
    return actual, predicted


def mase(y_true, y_pred, y_naive):
    """Mean absolute error divided by the seasonal-naive error on the same points."""
    actual, predicted = _columns(y_true, y_pred)
    naive = np.asarray(y_naive, dtype=float).reshape(-1)
    if len(naive) != len(actual):
        raise ValueError("Naive forecast length differs from the actuals")
    scale = float(np.mean(np.abs(naive - actual)))
    if scale == 0:
        return float("nan")
    return float(np.mean(np.abs(predicted - actual)) / scale)


def regression_metrics(y_true, y_pred, y_naive=None):
    """Return RMSE, MAE and MAPE (percent, over nonzero targets).

    When `y_naive` is given, also return MASE against that seasonal naive.
    """
    actual, predicted = _columns(y_true, y_pred)
    error = predicted - actual
    nonzero = actual != 0
    mape = (
        float(np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100.0)
        if nonzero.any()
        else float("nan")
    )
    metrics = {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "mape": mape,
    }
    if y_naive is not None:
        metrics["mase"] = mase(actual, predicted, y_naive)
    return metrics
