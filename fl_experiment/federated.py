"""FedAvg, FedProx and APFL updates for one rolling origin.

Aggregation is the sample-weighted average Flower uses for FedAvg. FedProx
changes only the client loss. The Flower FedProx strategy is not required to
apply the penalty; the penalty is computed here. mu = 0 leaves the loss as
plain mean squared error, so it matches FedAvg.
"""

import numpy as np
import torch

from fl_experiment.metrics import regression_metrics
from fl_experiment.model import RevenueModel

MU_GRID = (0.01, 0.1, 1.0)
ALPHA_STEPS = (0.1, 1.0)
ALPHA_INIT = 0.5
LOCAL_EPOCHS = 5
MAX_ROUNDS = 20
ROUND_PATIENCE = 5


def clip_alpha(value):
    """Keep a mixing weight inside the unit interval."""
    return float(min(1.0, max(0.0, value)))


def mix_parameters(local, global_params, alpha):
    """Return alpha * local + (1 - alpha) * global, layer by layer."""
    return [
        alpha * left + (1.0 - alpha) * right
        for left, right in zip(local, global_params)
    ]


def update_alpha(alpha, step, gradient):
    """One gradient step on the holdout loss, then clip to [0, 1]."""
    return clip_alpha(alpha - step * gradient)


def model_parameters(model):
    """Copy the state dict values in their iteration order."""
    return [value.detach().cpu().numpy().copy() for value in model.state_dict().values()]


def load_parameters(model, parameters):
    """Load arrays saved by `model_parameters`."""
    state = model.state_dict()
    loaded = {
        key: torch.as_tensor(value)
        for key, value in zip(state.keys(), parameters)
    }
    model.load_state_dict(loaded, strict=True)


def _new_model(n_features, settings):
    return RevenueModel(n_features, settings["hidden"], settings["dropout"])


def initial_parameters(n_features, settings):
    """Draw one shared initialization for every client at this origin."""
    torch.manual_seed(settings["seed"])
    return model_parameters(_new_model(n_features, settings))


def fedavg(updates):
    """Sample-weighted parameter average. `updates` is (count, parameters)."""
    total = float(sum(count for count, _ in updates))
    width = len(updates[0][1])
    return [
        sum(count * params[index] for count, params in updates) / total
        for index in range(width)
    ]


def local_update(start, client, n_features, settings, mu):
    """Full-batch Adam on the fit windows, plus the FedProx penalty when mu > 0."""
    model = _new_model(n_features, settings)
    load_parameters(model, start)
    anchors = [param.detach().clone() for param in model.parameters()]
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]
    )
    loss_fn = torch.nn.MSELoss()
    inputs = torch.from_numpy(client["fit"][0])
    target = torch.from_numpy(client["fit"][1])
    model.train()
    for _ in range(settings["local_epochs"]):
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), target)
        if mu:
            penalty = sum(
                torch.sum((param - anchor) ** 2)
                for param, anchor in zip(model.parameters(), anchors)
            )
            loss = loss + (mu / 2.0) * penalty
        loss.backward()
        optimizer.step()
    return model_parameters(model)


def alpha_gradient(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        local, global_params, alpha, client, n_features, settings):
    """Derivative of the holdout loss with respect to the mixing weight."""
    model = _new_model(n_features, settings)
    load_parameters(model, mix_parameters(local, global_params, alpha))
    model.eval()
    loss = torch.nn.MSELoss()(
        model(torch.from_numpy(client["early_stop"][0])),
        torch.from_numpy(client["early_stop"][1]),
    )
    loss.backward()
    dot = 0.0
    for param, left, right in zip(model.parameters(), local, global_params):
        gap = torch.as_tensor(left - right, dtype=param.dtype)
        dot += float(torch.sum(param.grad * gap))
    return dot


def predict_crore(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        parameters, inputs, y_mean, y_std, n_features, settings):
    """Decode one normalized block back to INR crore."""
    model = _new_model(n_features, settings)
    load_parameters(model, parameters)
    model.eval()
    with torch.no_grad():
        normalized = model(torch.from_numpy(inputs)).numpy().reshape(-1)
    return np.expm1(normalized * y_std + y_mean)


def client_holdout_rmse(parameters, client, n_features, settings):
    """One state's holdout RMSE in INR crore."""
    predicted = predict_crore(
        parameters, client["early_stop"][0], client["y_mean"], client["y_std"],
        n_features, settings,
    )
    actual = np.expm1(
        client["early_stop"][1].reshape(-1) * client["y_std"] + client["y_mean"]
    )
    return regression_metrics(actual, predicted)["rmse"]


def client_origin_prediction(parameters, client, n_features, settings):
    """One state's forecast for its origin month, in INR crore."""
    predicted = predict_crore(
        parameters, client["origin"][0], client["y_mean"], client["y_std"],
        n_features, settings,
    )
    return float(predicted[0])


def _holdout_rmse(scored, clients, n_features, settings):
    """Mean of the per-state holdout RMSE, in INR crore."""
    return float(np.mean([
        client_holdout_rmse(parameters, client, n_features, settings)
        for parameters, client in zip(scored, clients)
    ]))


def _copy_stack(parameters):
    return [layer.copy() for layer in parameters]


def client_step(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        global_params, client, alpha, n_features, settings, job):
    """Fit one client and, for APFL, move alpha on the holdout."""
    local = local_update(global_params, client, n_features, settings, job["mu"])
    if not job["personalize"]:
        return local, None, alpha
    gradient = alpha_gradient(
        local, global_params, alpha, client, n_features, settings
    )
    alpha = update_alpha(alpha, job["alpha_step"], gradient)
    return local, mix_parameters(local, global_params, alpha), alpha


def _round(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        global_params, clients, alphas, n_features, settings, job):
    """One communication round. Returns the new global model and scored copies."""
    updates, scored, next_alphas = [], [], []
    for client, alpha in zip(clients, alphas):
        local, personal, alpha = client_step(
            global_params, client, alpha, n_features, settings, job
        )
        updates.append((len(client["fit"][0]), local))
        scored.append(personal)
        next_alphas.append(alpha)
    aggregated = fedavg(updates)
    if job["personalize"]:
        saved = [_copy_stack(item) for item in scored]
    else:
        saved = [_copy_stack(aggregated) for _ in clients]
    holdout = _holdout_rmse(saved, clients, n_features, settings)
    return aggregated, next_alphas, saved, holdout


def run_origin(clients, settings, job):
    """Train at one origin and score its target month with the best holdout round.

    `job` carries mu, personalize, alpha_step and alpha_init. The target month
    is not read while choosing the round.
    """
    n_features = int(clients[0]["fit"][0].shape[-1])
    global_params = initial_parameters(n_features, settings)
    alphas = [job["alpha_init"] for _ in clients]
    best = {"holdout_rmse": float("inf"), "round": 0, "params": None, "alphas": []}
    stale, rounds_run = 0, 0
    for round_index in range(1, settings["max_rounds"] + 1):
        rounds_run = round_index
        global_params, alphas, scored, holdout = _round(
            global_params, clients, alphas, n_features, settings, job
        )
        if holdout < best["holdout_rmse"] - 1e-9:
            best = {
                "holdout_rmse": holdout,
                "round": round_index,
                "params": scored,
                "alphas": list(alphas),
            }
            stale = 0
        else:
            stale += 1
            if stale >= settings["round_patience"]:
                break
    rows = _score_origin(clients, best["params"], n_features, settings)
    info = {
        "rounds_run": rounds_run,
        "selected_round": best["round"],
        "alphas": best["alphas"],
    }
    return rows, info


def origin_row(client, predicted):
    """Prediction row for one state's origin month."""
    return {
        "state_id": client["state_id"],
        "target_month": client["target_month"],
        "actual": float(client["actual"]),
        "predicted": float(predicted),
        "naive": client["naive"],
        "fallback": "",
    }


def _score_origin(clients, scored, n_features, settings):
    return [
        origin_row(client, client_origin_prediction(parameters, client, n_features, settings))
        for client, parameters in zip(clients, scored)
    ]


def make_job(kind, value=0.0, alpha_init=ALPHA_INIT):
    """Method settings shared by the in-process and Flower paths."""
    if kind not in {"fedavg", "fedprox", "apfl"}:
        raise ValueError("kind must be 'fedavg', 'fedprox' or 'apfl'")
    return {
        "kind": kind,
        "mu": float(value) if kind == "fedprox" else 0.0,
        "personalize": kind == "apfl",
        "alpha_step": float(value) if kind == "apfl" else 0.0,
        "alpha_init": float(alpha_init),
    }


def fedavg_origin(clients, settings):
    """FedAvg, which is FedProx with mu = 0 and no personalization."""
    return run_origin(clients, settings, make_job("fedavg"))


def fedprox_origin(clients, settings, mu):
    """FedProx. mu = 0 is required to match `fedavg_origin` on the same seed."""
    return run_origin(clients, settings, make_job("fedprox", mu))


def apfl_origin(clients, settings, alpha_step, alpha_init=ALPHA_INIT):
    """APFL. The scored parameters are the mixture, not the averaged local model."""
    return run_origin(clients, settings, make_job("apfl", alpha_step, alpha_init))
