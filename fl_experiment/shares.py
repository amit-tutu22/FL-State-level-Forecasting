"""FH-PFL v2: federated top-down shares on the frozen rolling origins.

Specification: docs/methods/fh_pfl_v2.md. Each state keeps a private GRU that
scores its log share of the national series N_t, the sum of the federation
states. States are coupled only through per-month sums across states. Those
sums are computed here in-process as a stand-in for secure summation; no
cryptography is applied and no privacy guarantee is claimed.
"""

import math

import numpy as np
import torch

from fl_experiment.baselines import forecast_sarima, seasonal_naive, select_state_order
from fl_experiment.federated import (
    initial_parameters,
    load_parameters,
    model_parameters,
    origin_row,
)
from fl_experiment.folds import column_index
from fl_experiment.model import RevenueModel
from fl_experiment.state_data import _stack, _training_scales, shift_month, window_targets

TOP_DOWN_MONTHS = 12
V2 = "fh_pfl_v2"
MODEL_NAMES = {
    "v2": V2,
    "a1": f"{V2}_a1_independent",
    "a2": f"{V2}_a2_posthoc",
    "a3": f"{V2}_a3_top_down",
    "a4": f"{V2}_a4_naive_anchor",
    "a5": f"{V2}_a5_no_national_input",
    "oracle": f"{V2}_oracle_anchor",
}
TRAINED = {
    "v2": {"coupled": True, "national_input": True},
    "a1": {"coupled": False, "national_input": True},
    "a5": {"coupled": True, "national_input": False},
}
ANCHOR_FALLBACK = "national_seasonal_naive"


def federation_months(group):
    """Months in which every state of the federation has a row."""
    shared = set(group[0][1]["months"])
    for _, state, _ in group[1:]:
        shared &= set(state["months"])
    return sorted(shared)


def national_series(group, months, before):
    """N_t, the sum of the federation states, for shared months strictly before `before`."""
    values = [dict(zip(state["months"], state["targets"])) for _, state, _ in group]
    return {
        month: float(sum(series[month] for series in values))
        for month in months
        if month < before
    }


def national_state(group, before):
    """The national series shaped like a fold state, for the SARIMA helpers."""
    series = national_series(group, federation_months(group), before)
    return {"months": list(series), "targets": list(series.values())}


def national_inputs(group, months, gst_column):
    """log1p of the summed lagged GST input, the public national input column."""
    rows = [dict(zip(state["months"], state["inputs"])) for _, state, _ in group]
    return {
        month: math.log1p(sum(math.expm1(row[month][gst_column]) for row in rows))
        for month in months
    }


def same_federation(groups):
    """Stop unless every origin month has the same set of states."""
    members = {month: sorted(state_id for state_id, _, _ in group)
               for month, group in groups.items()}
    first = next(iter(members.values()))
    if any(item != first for item in members.values()):
        raise ValueError("The federation changes between origins; N_t would be redefined")
    return first


def national_anchors(groups, val_start, grid, min_train, maxiter):
    """SARIMA anchor per origin month, with the order frozen before `val_start`.

    Returns (anchors, chosen order). Each anchor also carries the seasonal
    naive (ablation A4) and the actual N_m (oracle only).
    """
    first = groups[min(groups)]
    chosen = select_state_order(
        national_state(first, val_start), val_start, grid, min_train, maxiter
    )
    anchors = {}
    for month, group in groups.items():
        state = national_state(group, month)
        forecast, fell_back = forecast_sarima(state, month, chosen, min_train, maxiter)
        anchors[month] = {
            "sarima": float(forecast),
            "fallback": bool(fell_back),
            "naive": float(seasonal_naive(state, month)),
            "actual": float(sum(origin["actual"] for _, _, origin in group)),
        }
    return anchors, chosen


def _share_inputs(state, public):
    """GST input, then the national column when used, then the month encoding."""
    own = dict(zip(state["months"], state["inputs"]))
    columns, national = public["columns"], public["national_inputs"]
    return {
        month: [own[month][index] for index in columns[:-2]]
        + ([national[month]] if national else [])
        + [own[month][index] for index in columns[-2:]]
        for month in public["months"]
    }


def _split_months(public, month):
    """Fit and holdout target months whose windows lie in the shared months."""
    usable = window_targets(public["months"], public["window"])
    if month not in usable:
        raise ValueError(f"No input window at {month}")
    fit_end = shift_month(month, -public["holdout"])
    return (
        [item for item in usable if item < fit_end],
        [item for item in usable if fit_end <= item < month],
        fit_end,
    )


def _share_client(state_id, state, origin, public):
    """One state's private arrays at one origin. Nothing reads the origin month's target."""
    month = origin["target_month"]
    targets = dict(zip(state["months"], state["targets"]))
    inputs = _share_inputs(state, public)
    log_share = {
        item: math.log(targets[item] / total) for item, total in public["national"].items()
    }
    fit_months, early_months, fit_end = _split_months(public, month)
    x_mean, x_std, _, _ = _training_scales(inputs, log_share, fit_end)
    fit_target = np.asarray([log_share[item] for item in fit_months], dtype=float)

    def block(target_months):
        raw = _stack(inputs, target_months, public["window"], len(x_mean))
        return ((raw - x_mean) / x_std).astype(np.float32)

    return {
        "state_id": state_id,
        "target_month": month,
        "actual": float(origin["actual"]),
        "naive": origin["seasonal_naive"],
        "fit": block(fit_months),
        "log_share": fit_target,
        "early": block(early_months),
        "early_actual": np.asarray([targets[item] for item in early_months], dtype=float),
        "early_national": np.asarray(
            [public["national"][item] for item in early_months], dtype=float
        ),
        "origin": block([month]),
        "share_mean": float(fit_target.mean()),
        "share_std": float(fit_target.std()) if fit_target.std() > 1e-8 else 1.0,
        "fit_months": fit_months,
        "early_months": early_months,
    }


def share_clients(group, folds, national_input=True):
    """Per-state share arrays for one origin month. Every state sees the same fit months."""
    protocol = folds["protocol"]
    month = group[0][2]["target_month"]
    columns = column_index(folds["feature_names"], "gst_only")
    if len(columns) != 3:
        raise ValueError("FH-PFL v2 expects exactly one GST input column")
    months = [item for item in federation_months(group) if item <= month]
    public = {
        "window": protocol["window"],
        "holdout": protocol["holdout_months"],
        "columns": columns,
        "months": months,
        "national": national_series(group, months, month),
        "national_inputs": national_inputs(group, months, columns[0]) if national_input else {},
    }
    clients = [
        _share_client(state_id, state, origin, public) for state_id, state, origin in group
    ]
    if any(client["fit_months"] != clients[0]["fit_months"] for client in clients):
        raise ValueError("States disagree on the fit months")
    return clients


def _new_model(n_features, settings):
    return RevenueModel(n_features, settings["hidden"], settings["dropout"])


def decode_scores(output, client):
    """g = sigma_i * out + m_i, in float64."""
    return output.reshape(-1).double() * client["share_std"] + client["share_mean"]


def log_scores(parameters, client, block, settings):
    """Log-share scores g for one block ('fit', 'early' or 'origin'), in eval mode."""
    model = _new_model(int(client[block].shape[-1]), settings)
    load_parameters(model, parameters)
    model.eval()
    with torch.no_grad():
        return decode_scores(model(torch.from_numpy(client[block])), client).numpy()


def surrogate(scores, log_share, offset, total_error):
    """S_i. With offset and total_error at zero it is A1's squared log-share error."""
    gap = scores - offset
    return torch.mean((gap - log_share) ** 2 - 2.0 * total_error * torch.exp(gap))


def local_step(parameters, client, broadcast, settings):
    """Full-batch Adam on S_i with the round-start offset c_t and total error E_t."""
    model = _new_model(int(client["fit"].shape[-1]), settings)
    load_parameters(model, parameters)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]
    )
    inputs = torch.from_numpy(client["fit"])
    target = torch.from_numpy(client["log_share"])
    offset = torch.as_tensor(broadcast["offset"], dtype=torch.float64)
    total = torch.as_tensor(broadcast["total_error"], dtype=torch.float64)
    model.train()
    for _ in range(settings["local_epochs"]):
        optimizer.zero_grad()
        surrogate(decode_scores(model(inputs), client), target, offset, total).backward()
        optimizer.step()
    return model_parameters(model)


def score_upload(parameters, client, settings):
    """First upload: exp(g_t) for every fit month."""
    return np.exp(log_scores(parameters, client, "fit", settings))


def error_upload(parameters, client, offset, settings):
    """Second upload: e_t = g_t - c_t - log s_t for every fit month."""
    return log_scores(parameters, client, "fit", settings) - offset - client["log_share"]


def secure_sum(uploads):
    """Element-wise sum across states. An in-process stand-in for secure aggregation."""
    return np.sum(np.stack([np.asarray(item, dtype=float) for item in uploads]), axis=0)


def holdout_rmse(early_scores, normaliser, client):
    """One state's holdout RMSE in INR crore, using the actual N_t of each holdout month."""
    predicted = np.exp(early_scores) / normaliser * client["early_national"]
    return float(np.sqrt(np.mean((predicted - client["early_actual"]) ** 2)))


def zero_broadcast(client):
    """c_t = E_t = 0: the uncoupled update of ablation A1."""
    width = len(client["log_share"])
    return {"offset": np.zeros(width), "total_error": np.zeros(width)}


def coupling(parameters, clients, settings):
    """Two secure sums give c_t = log Z_t and E_t for the coming local step."""
    offset = np.log(secure_sum([
        score_upload(params, client, settings) for params, client in zip(parameters, clients)
    ]))
    total = secure_sum([
        error_upload(params, client, offset, settings)
        for params, client in zip(parameters, clients)
    ])
    return {"offset": offset, "total_error": total}


def joint_objective(scores, log_shares):
    """J: equal-state mean squared log-share error, for the gradient check."""
    offset = torch.log(torch.sum(torch.exp(torch.stack(scores)), dim=0))
    errors = [torch.mean((score - offset - share) ** 2)
              for score, share in zip(scores, log_shares)]
    return torch.mean(torch.stack(errors))


def _round_scores(parameters, clients, settings, coupled):
    early = [log_scores(params, client, "early", settings)
             for params, client in zip(parameters, clients)]
    origin = np.asarray([log_scores(params, client, "origin", settings)[0]
                         for params, client in zip(parameters, clients)])
    normaliser = secure_sum([np.exp(item) for item in early]) if coupled else 1.0
    rmse = float(np.mean([
        holdout_rmse(scores, normaliser, client) for scores, client in zip(early, clients)
    ]))
    return rmse, origin


def run_share_origin(clients, settings, coupled):
    """Train at one origin; keep the round with the lowest equal-state holdout RMSE.

    Returns the origin-month log-share scores of that round. Each state starts
    from the same seeded initialisation and keeps its own parameters after.
    """
    start = initial_parameters(int(clients[0]["fit"].shape[-1]), settings)
    parameters = [[layer.copy() for layer in start] for _ in clients]
    best = {"holdout_rmse": float("inf"), "round": 0, "origin_scores": None}
    stale, rounds_run = 0, 0
    for round_index in range(1, settings["max_rounds"] + 1):
        rounds_run = round_index
        broadcast = coupling(parameters, clients, settings) if coupled else None
        parameters = [
            local_step(params, client, broadcast or zero_broadcast(client), settings)
            for params, client in zip(parameters, clients)
        ]
        rmse, origin = _round_scores(parameters, clients, settings, coupled)
        improved = rmse < best["holdout_rmse"] - 1e-9
        if improved:
            best = {"holdout_rmse": rmse, "round": round_index, "origin_scores": origin}
        stale = 0 if improved else stale + 1
        if stale >= settings["round_patience"]:
            break
    best["rounds_run"] = rounds_run
    return best


def forecast_shares(origin_scores, normalise):
    """exp(g_m), divided by the secure sum Z_m when `normalise`."""
    weights = np.exp(np.asarray(origin_scores, dtype=float))
    return weights / weights.sum() if normalise else weights


def _rows(name, clients, shares, anchor_value, fallback=""):
    rows = []
    for client, share in zip(clients, shares):
        row = origin_row(client, float(share) * anchor_value)
        row["model"] = name
        row["fallback"] = fallback
        rows.append(row)
    return rows


def trained_rows(kind, clients, origin_scores, anchor):
    """Prediction rows for one trained variant at one origin."""
    fallback = ANCHOR_FALLBACK if anchor["fallback"] else ""
    if kind == "a1":
        return (
            _rows(MODEL_NAMES["a1"], clients, forecast_shares(origin_scores, False),
                  anchor["sarima"], fallback)
            + _rows(MODEL_NAMES["a2"], clients, forecast_shares(origin_scores, True),
                    anchor["sarima"], fallback)
        )
    shares = forecast_shares(origin_scores, True)
    if kind == "a5":
        return _rows(MODEL_NAMES["a5"], clients, shares, anchor["sarima"], fallback)
    return (
        _rows(MODEL_NAMES["v2"], clients, shares, anchor["sarima"], fallback)
        + _rows(MODEL_NAMES["a4"], clients, shares, anchor["naive"])
        + _rows(MODEL_NAMES["oracle"], clients, shares, anchor["actual"])
    )


def top_down_rows(clients, anchor):
    """A3: the SARIMA anchor times the mean share of the last 12 fit months."""
    shares = [float(np.mean(np.exp(client["log_share"][-TOP_DOWN_MONTHS:])))
              for client in clients]
    fallback = ANCHOR_FALLBACK if anchor["fallback"] else ""
    return _rows(MODEL_NAMES["a3"], clients, shares, anchor["sarima"], fallback)
