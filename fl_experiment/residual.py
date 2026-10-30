"""FH-PFL v3: federated personalized residual correction of SARIMA.

The forecast is base * exp(lambda * r_hat), where base is the state's
frozen-order SARIMA forecast and r_hat is a GRU forecast of SARIMA's log
error. See docs/methods/fh_pfl_v3.md and fh_pfl_v3_methodology.md.

Two SARIMA fits per state and origin month m:
- the residual fit, on months before the holdout, whose one-step filtered
  predictions (parameters fixed through the holdout) give the residual
  targets;
- the forecast fit, on months before m, which is the SARIMA baseline itself.
Neither reads month m.
"""

import math
import warnings

import numpy as np
import torch

from fl_experiment.baselines import fit_sarima, forecast_sarima, select_state_order
from fl_experiment.federated import (
    decode_crore,
    load_parameters,
    make_job,
    origin_row,
    run_origin,
)
from fl_experiment.folds import _sliced_inputs, suffix_values
from fl_experiment.local_gru import train_arrays
from fl_experiment.model import RevenueModel
from fl_experiment.state_data import _stack, shift_month, window_targets

BURN_IN = 13
LAMBDAS = (0.0, 0.5, 1.0)
RIDGE_PENALTY = 1.0
INPUTS = ("gst_ewb", "gst_only")
SARIMA_FALLBACK = "seasonal_naive"
NO_RESIDUAL = "no_residual_fit"
MODEL_NAMES = {
    "v3": "fh_pfl_v3",
    "b0": "fh_pfl_v3_b0_sarima",
    "b1": "fh_pfl_v3_b1_local",
    "b2": "fh_pfl_v3_b2_fedavg",
    "b3": "fh_pfl_v3_b3_ridge",
    "b4": "fh_pfl_v3_b4_gst_only",
    "b5": "fh_pfl_v3_b5_full_correction",
}
TRAINED = {
    "v3": {"trainer": "apfl", "inputs": "gst_ewb"},
    "b1": {"trainer": "local", "inputs": "gst_ewb"},
    "b2": {"trainer": "fedavg", "inputs": "gst_ewb"},
    "b4": {"trainer": "apfl", "inputs": "gst_only"},
}


def residual_columns(feature_names, inputs):
    """Input columns for the residual GRU. The last two are the month encoding.

    `gst_ewb` keeps every input except CPI; `gst_only` keeps the GST inputs.
    """
    width = len(feature_names) + 2
    if inputs == "gst_ewb":
        keep = [index for index, name in enumerate(feature_names) if "cpi" not in name]
    elif inputs == "gst_only":
        keep = [index for index, name in enumerate(feature_names) if "gst" in name]
    else:
        raise ValueError(f"inputs must be one of {INPUTS}")
    return keep + [width - 2, width - 1]


def frozen_orders(groups, protocol, grid, maxiter):
    """Each state's SARIMA order, chosen by AIC before the validation year."""
    orders = {}
    for group in groups.values():
        for state_id, state, _ in group:
            if state_id not in orders:
                orders[state_id] = select_state_order(
                    state, protocol["val_start"], grid, protocol["min_train"], maxiter
                )
    return orders


def filtered_base(state, month, chosen, protocol, maxiter):
    """Residual fit: one-step filtered SARIMA predictions, keyed by month.

    Parameters are estimated on the contiguous run ending before the holdout,
    then held fixed while the filter runs through the holdout months. Each
    prediction for month t uses observations before t only. The first
    BURN_IN months are dropped. Returns {} when the fit is not possible.
    """
    fit_end = shift_month(month, -protocol["holdout_months"])
    last = shift_month(fit_end, -1)
    values = suffix_values(state, last)
    if not chosen["order"] or len(values) < protocol["min_train"]:
        return {}
    targets = dict(zip(state["months"], state["targets"]))
    holdout = [shift_month(fit_end, offset) for offset in range(protocol["holdout_months"])]
    try:
        result, _ = fit_sarima(values, chosen["order"], chosen["seasonal_order"], maxiter)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            extended = result.append([targets[item] for item in holdout], refit=False)
    except Exception:  # pylint: disable=broad-exception-caught
        return {}
    fitted = np.asarray(extended.fittedvalues, dtype=float)
    start = shift_month(last, 1 - len(values))
    months = [shift_month(start, offset) for offset in range(len(fitted))]
    return {
        item: float(value)
        for item, value in zip(months[BURN_IN:], fitted[BURN_IN:])
        if math.isfinite(value) and value > 0
    }


def sarima_bases(group, orders, protocol, maxiter):
    """Both SARIMA fits for every state of one origin month."""
    bases = {}
    for state_id, state, origin in group:
        month = origin["target_month"]
        forecast, fell_back = forecast_sarima(
            state, month, orders[state_id], protocol["min_train"], maxiter
        )
        bases[state_id] = {
            "filtered": filtered_base(state, month, orders[state_id], protocol, maxiter),
            "forecast": float(forecast),
            "fallback": bool(fell_back),
        }
    return bases


def _windows(inputs, months, window, log_bases):
    """Input windows plus the target month's SARIMA log-forecast at every step."""
    width = len(next(iter(inputs.values())))
    block = _stack(inputs, months, window, width)
    column = np.repeat(np.asarray(log_bases, dtype=float).reshape(-1, 1, 1), window, axis=1)
    return np.concatenate([block, column], axis=2)


def _safe(value):
    return float(value) if value > 1e-8 else 1.0


def _residual_months(state, month, filtered, protocol):
    """Fit and holdout target months, or None when the holdout is incomplete."""
    fit_end = shift_month(month, -protocol["holdout_months"])
    usable = set(window_targets(state["months"], protocol["window"]))
    fit = [item for item in sorted(filtered) if item < fit_end and item in usable]
    early = [shift_month(fit_end, offset) for offset in range(protocol["holdout_months"])]
    if len(fit) < 2 or any(item not in filtered or item not in usable for item in early):
        return None
    if month in fit or month in early:
        raise ValueError("Origin month leaked into the residual targets")
    return fit, early


def residual_client(group_item, base, columns, protocol):
    """Client dict in the shape `federated.py` consumes, or None.

    Targets are normalized log residuals log(y / base). `base_early` and
    `base_origin` let the shared code decode to INR crore.
    """
    state_id, state, origin = group_item
    month = origin["target_month"]
    months = _residual_months(state, month, base["filtered"], protocol)
    if months is None or base["forecast"] <= 0:
        return None
    targets = dict(zip(state["months"], state["targets"]))
    residuals = [
        np.log([targets[item] / base["filtered"][item] for item in block]) for block in months
    ]
    r_mean, r_std = float(residuals[0].mean()), _safe(residuals[0].std())
    scaled = _scaled_windows(
        _sliced_inputs(state, columns), months, month, base, protocol["window"]
    )

    def target(values):
        return ((values - r_mean) / r_std).reshape(-1, 1).astype(np.float32)

    return {
        "state_id": state_id,
        "target_month": month,
        "actual": origin["actual"],
        "naive": origin["seasonal_naive"],
        "fit": (scaled[0], target(residuals[0])),
        "early_stop": (scaled[1], target(residuals[1])),
        "origin": (scaled[2], np.zeros((1, 1), dtype=np.float32)),
        "y_mean": r_mean,
        "y_std": r_std,
        "base_early": np.asarray([base["filtered"][item] for item in months[1]], dtype=float),
        "base_origin": np.asarray([base["forecast"]], dtype=float),
        "sarima_fallback": base["fallback"],
    }


def _scaled_windows(inputs, months, month, base, window):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Fit, holdout and origin windows, standardized on the fit windows."""
    filtered = base["filtered"]
    raw = [
        _windows(inputs, block, window, np.log([filtered[item] for item in block]))
        for block in months
    ]
    raw.append(_windows(inputs, [month], window, [math.log(base["forecast"])]))
    flat = raw[0].reshape(-1, raw[0].shape[-1])
    spread = flat.std(axis=0)
    x_mean, x_std = flat.mean(axis=0), np.where(spread < 1e-8, 1.0, spread)
    return [((block - x_mean) / x_std).astype(np.float32) for block in raw]


def residual_clients(group, bases, columns, protocol):
    """(state id -> client or None) for one origin month."""
    return {
        item[0]: residual_client(item, bases[item[0]], columns, protocol)
        for item in group
    }


def choose_lambda(early_residual, client):
    """Shrinkage from LAMBDAS with the lowest holdout RMSE in crore; ties go to the smaller."""
    base = client["base_early"]
    actual = decode_crore(client["early_stop"][1], client["y_mean"], client["y_std"], base)

    def score(value):
        predicted = base * np.exp(value * np.asarray(early_residual, dtype=float))
        return round(float(np.sqrt(np.mean((predicted - actual) ** 2))), 9)

    return min(LAMBDAS, key=lambda value: (score(value), value))


def corrected(client, origin_residual, value):
    """The v3 forecast for the origin month, in INR crore."""
    return float(client["base_origin"][0] * math.exp(value * float(origin_residual)))


def gru_residuals(parameters, client, settings):
    """Decoded log residual forecasts (holdout block, origin month) from GRU weights."""
    model = RevenueModel(client["fit"][0].shape[-1], settings["hidden"], settings["dropout"])
    load_parameters(model, parameters)
    model.eval()
    with torch.no_grad():
        blocks = [
            model(torch.from_numpy(client[key][0])).numpy().reshape(-1)
            for key in ("early_stop", "origin")
        ]
    return [block * client["y_std"] + client["y_mean"] for block in blocks]


def ridge_residuals(client, penalty=RIDGE_PENALTY):
    """Per-state ridge on the last window step, with an unpenalized intercept."""
    inputs = client["fit"][0][:, -1, :].astype(float)
    target = client["fit"][1].reshape(-1).astype(float)
    centre = inputs.mean(axis=0)
    centred = inputs - centre
    weights = np.linalg.solve(
        centred.T @ centred + penalty * np.eye(centred.shape[1]),
        centred.T @ (target - target.mean()),
    )

    def predict(key):
        last = client[key][0][:, -1, :].astype(float)
        normalized = (last - centre) @ weights + target.mean()
        return normalized * client["y_std"] + client["y_mean"]

    return [predict("early_stop"), predict("origin")]


def federated_residuals(clients, settings, job):
    """Train with FedAvg or APFL; return each client's residual forecasts."""
    _, info = run_origin(clients, settings, job)
    residuals = [
        gru_residuals(parameters, client, settings)
        for parameters, client in zip(info["params"], clients)
    ]
    return residuals, {"selected_round": info["selected_round"], "rounds_run": info["rounds_run"]}


def local_residuals(clients, settings, state_seeds):
    """One GRU per state, early-stopped on its holdout as the local GRU baseline."""
    residuals = []
    for client in clients:
        seeded = dict(settings, seed=settings["seed"] + state_seeds[client["state_id"]])
        trained, _ = train_arrays(
            torch.from_numpy(client["fit"][0]),
            torch.from_numpy(client["fit"][1]),
            torch.from_numpy(client["early_stop"][0]),
            torch.from_numpy(client["early_stop"][1]),
            client["fit"][0].shape[-1],
            seeded,
        )
        trained.eval()
        with torch.no_grad():
            blocks = [
                trained(torch.from_numpy(client[key][0])).numpy().reshape(-1)
                for key in ("early_stop", "origin")
            ]
        residuals.append([block * client["y_std"] + client["y_mean"] for block in blocks])
    return residuals


def _tagged(model, client, predicted):
    row = origin_row(client, predicted)
    row["model"] = model
    row["fallback"] = SARIMA_FALLBACK if client["sarima_fallback"] else ""
    return row


def shrunk_rows(model, clients, residuals, fixed=None):
    """Rows with lambda chosen on the holdout, or a fixed lambda. Returns rows and lambdas."""
    rows, lambdas = [], {}
    for client, (early, origin) in zip(clients, residuals):
        value = choose_lambda(early, client) if fixed is None else fixed
        rows.append(_tagged(model, client, corrected(client, origin[0], value)))
        lambdas[client["state_id"]] = value
    return rows, lambdas


def sarima_only_rows(model, group, bases, missing):
    """SARIMA rows for `missing` states, which have no residual fit at this origin."""
    rows = []
    for state_id, _, origin in group:
        if state_id not in missing:
            continue
        rows.append({
            "model": model,
            "state_id": state_id,
            "target_month": origin["target_month"],
            "actual": float(origin["actual"]),
            "predicted": bases[state_id]["forecast"],
            "naive": origin["seasonal_naive"],
            "fallback": NO_RESIDUAL,
        })
    return rows


def frozen_apfl(selection):
    """The frozen APFL rule (and FedAvg) from the selection file."""
    return {
        "apfl": make_job("apfl", selection["apfl_alpha_step"], selection["apfl_alpha_init"]),
        "fedavg": make_job("fedavg"),
    }
