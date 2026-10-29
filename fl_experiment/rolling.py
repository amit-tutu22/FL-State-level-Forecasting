"""Local GRU, pooled GRU and federated forecasts on shared rolling origins.

Every method retrains from its seed at each origin. Scaling and fitting use
only months before the origin, and the origin month is not used to stop.
"""

import numpy as np
import torch

from fl_experiment.federated import apfl_origin, fedavg_origin, fedprox_origin
from fl_experiment.folds import column_index, origin_arrays, pooled_scales, shift_month
from fl_experiment.local_gru import train_arrays
from fl_experiment.metrics import prediction_row


def origin_months(folds, split):
    """Sorted target months that have at least one eligible origin."""
    found = set()
    for state in folds["states"].values():
        for origin in state["origins"]:
            if _eligible(origin, split):
                found.add(origin["target_month"])
    return sorted(found)


def states_for_month(folds, month, split):
    """Eligible (state id, state, origin) triples for one target month."""
    found = []
    for state_id, state in folds["states"].items():
        for origin in state["origins"]:
            if origin["target_month"] == month and _eligible(origin, split):
                found.append((state_id, state, origin))
    return found


def _eligible(origin, split):
    if split == "val":
        return bool(origin["scored"])
    if split == "test":
        return origin["split"] == "test" and origin["skip_reason"] == ""
    raise ValueError("split must be 'val' or 'test'")


def _row(model, seed, state_id, origin, predicted):
    """Prediction row tagged with the seed that trained it."""
    row = prediction_row(model, state_id, origin, predicted)
    row["seed"] = seed
    return row


def _decode(model, block, y_mean, y_std):
    with torch.no_grad():
        normalized = model(torch.from_numpy(block)).numpy().reshape(-1)[0]
    return float(np.expm1(normalized * y_std + y_mean))


def local_rows(folds, feature_set, settings, split):
    """One GRU per state. The same seed is reused at every origin of that state."""
    protocol = folds["protocol"]
    columns = column_index(folds["feature_names"], feature_set)
    rows = []
    for index, (state_id, state) in enumerate(folds["states"].items()):
        seeded = dict(settings, seed=settings["seed"] + index)
        for origin in state["origins"]:
            if not _eligible(origin, split):
                continue
            arrays = origin_arrays(
                state, origin, protocol["window"], protocol["holdout_months"], columns
            )
            trained, _ = train_arrays(
                torch.from_numpy(arrays["fit"][0]),
                torch.from_numpy(arrays["fit"][1]),
                torch.from_numpy(arrays["early_stop"][0]),
                torch.from_numpy(arrays["early_stop"][1]),
                arrays["fit"][0].shape[-1],
                seeded,
            )
            predicted = _decode(trained, arrays["origin"][0], arrays["y_mean"], arrays["y_std"])
            rows.append(_row(
                f"local_gru_{feature_set}", settings["seed"], state_id, origin, predicted
            ))
    return rows


def _pooled_blocks(group, month, feature_set, folds):
    protocol = folds["protocol"]
    columns = column_index(folds["feature_names"], feature_set)
    holdout = protocol["holdout_months"]
    scales = pooled_scales(
        [state for _, state, _ in group], shift_month(month, -holdout), columns
    )
    blocks = []
    for state_id, state, origin in group:
        arrays = origin_arrays(
            state, origin, protocol["window"], holdout, columns, scales
        )
        blocks.append((state_id, origin, arrays))
    return blocks


def pooled_rows(folds, feature_set, settings, split):
    """One GRU per origin, fit on every eligible state's windows with one scaler."""
    rows = []
    for month in origin_months(folds, split):
        group = states_for_month(folds, month, split)
        blocks = _pooled_blocks(group, month, feature_set, folds)
        trained, _ = train_arrays(
            torch.from_numpy(np.concatenate([item[2]["fit"][0] for item in blocks])),
            torch.from_numpy(np.concatenate([item[2]["fit"][1] for item in blocks])),
            torch.from_numpy(np.concatenate([item[2]["early_stop"][0] for item in blocks])),
            torch.from_numpy(np.concatenate([item[2]["early_stop"][1] for item in blocks])),
            blocks[0][2]["fit"][0].shape[-1],
            settings,
        )
        for state_id, origin, arrays in blocks:
            predicted = _decode(trained, arrays["origin"][0], arrays["y_mean"], arrays["y_std"])
            rows.append(_row("pooled_gru", settings["seed"], state_id, origin, predicted))
    return rows


def federated_clients(group, feature_set, folds):
    """Per-state scaled arrays for one origin. Each state keeps its own scaler."""
    protocol = folds["protocol"]
    columns = column_index(folds["feature_names"], feature_set)
    clients = []
    for state_id, state, origin in group:
        arrays = origin_arrays(
            state, origin, protocol["window"], protocol["holdout_months"], columns
        )
        clients.append({
            "state_id": state_id,
            "target_month": origin["target_month"],
            "actual": origin["actual"],
            "naive": origin["seasonal_naive"],
            "fit": arrays["fit"],
            "early_stop": arrays["early_stop"],
            "origin": arrays["origin"],
            "y_mean": arrays["y_mean"],
            "y_std": arrays["y_std"],
        })
    return clients


def _tag(rows, model, seed):
    """Attach the method name and seed after a federated origin has been scored."""
    for row in rows:
        row["model"] = model
        row["seed"] = seed
    return rows


def federated_rows(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        folds, feature_set, settings, split, kind, value=0.0):
    """FedAvg, FedProx or APFL at every eligible origin of `split`."""
    runners = {
        "fedavg": lambda clients: fedavg_origin(clients, settings),
        "fedprox": lambda clients: fedprox_origin(clients, settings, value),
        "apfl": lambda clients: apfl_origin(clients, settings, value),
    }
    names = {
        "fedavg": "fedavg",
        "fedprox": f"fedprox_mu_{value:g}",
        "apfl": f"apfl_step_{value:g}",
    }
    rows = []
    for month in origin_months(folds, split):
        clients = federated_clients(states_for_month(folds, month, split), feature_set, folds)
        batch, _ = runners[kind](clients)
        rows.extend(batch)
    return _tag(rows, names[kind], settings["seed"])
