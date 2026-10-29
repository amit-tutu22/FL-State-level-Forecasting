"""FedAvg strategy and a standalone Flower server.

Standalone:
    python -m fl_experiment.server --rounds 5 --min-clients 10 --address 0.0.0.0:8080
"""

import argparse

import flwr as fl

METRIC_KEYS = ("rmse", "mae", "mape")


class MetricsRecorder:  # pylint: disable=too-few-public-methods
    """Aggregates client metrics and keeps the per-state values of every round."""

    def __init__(self):
        self.per_state = []

    def __call__(self, results):
        total = sum(count for count, _ in results)
        self.per_state.append(
            [dict(metrics, examples=count) for count, metrics in results]
        )
        if total == 0:
            return {}
        weighted = {
            key: sum(count * metrics[key] for count, metrics in results) / total
            for key in METRIC_KEYS
            if all(key in metrics for _, metrics in results)
        }
        equal = {
            f"{key}_equal_state": sum(metrics[key] for _, metrics in results) / len(results)
            for key in METRIC_KEYS
            if all(key in metrics for _, metrics in results)
        }
        return {**weighted, **equal}


def average_train_loss(results):
    """Sample-weighted mean of client training losses."""
    total = sum(count for count, _ in results)
    if total == 0:
        return {}
    return {"train_loss": sum(count * m["train_loss"] for count, m in results) / total}


def build_strategy(num_clients, recorder=None):
    """FedAvg with full participation; every round waits for all state clients."""
    return fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        fit_metrics_aggregation_fn=average_train_loss,
        evaluate_metrics_aggregation_fn=recorder or MetricsRecorder(),
    )


class OriginRecorder:
    """Per-round holdout scores for one rolling origin, with the patience rule.

    The kept round has the lowest equal-state holdout RMSE. Origin-month
    predictions are stored with each round and never used to choose one.
    """

    def __init__(self, patience, personalize):
        self.patience = patience
        self.personalize = personalize
        self.current_round = 0
        self.stale = 0
        self.stopped = False
        self.history = []
        self.best = {"holdout_rmse": float("inf"), "round": 0, "predictions": {}, "alphas": {}}

    def close_round(self, results):
        """Record one round's client scores and update the stopping state."""
        holdout = sum(metrics["holdout_rmse"] for _, metrics in results) / len(results)
        predictions = {metrics["state_id"]: metrics["prediction"] for _, metrics in results}
        alphas = {
            metrics["state_id"]: metrics["alpha"] for _, metrics in results if "alpha" in metrics
        }
        self.history.append({"round": self.current_round, "holdout_rmse": holdout})
        if holdout < self.best["holdout_rmse"] - 1e-9:
            self.best = {
                "holdout_rmse": holdout,
                "round": self.current_round,
                "predictions": predictions,
                "alphas": alphas,
            }
            self.stale = 0
        else:
            self.stale += 1
            if self.stale >= self.patience:
                self.stopped = True
        return {"holdout_rmse": holdout}

    def on_fit(self, results):
        """APFL scores its personalized mixture during fit."""
        return self.close_round(results) if self.personalize else {}

    def on_evaluate(self, results):
        """FedAvg and FedProx score the new global model."""
        return {} if self.personalize else self.close_round(results)


def _fit_round(recorder, configure, arguments):
    """Skip the round after early stopping."""
    if recorder.stopped:
        return []
    recorder.current_round = arguments[0]
    return configure(*arguments)


def _evaluate_round(recorder, configure, arguments):
    """APFL is scored in fit, so it needs no evaluation round."""
    if recorder.stopped or recorder.personalize:
        return []
    return configure(*arguments)


class OriginFedAvg(fl.server.strategy.FedAvg):
    """FedAvg for one rolling origin. Stops selecting clients after patience."""

    recorder = None

    def configure_fit(self, server_round, parameters, client_manager):
        arguments = (server_round, parameters, client_manager)
        return _fit_round(self.recorder, super().configure_fit, arguments)

    def configure_evaluate(self, server_round, parameters, client_manager):
        arguments = (server_round, parameters, client_manager)
        return _evaluate_round(self.recorder, super().configure_evaluate, arguments)


class OriginFedProx(fl.server.strategy.FedProx):
    """FedProx for one rolling origin. The strategy only sends `proximal_mu`."""

    recorder = None

    def configure_fit(self, server_round, parameters, client_manager):
        arguments = (server_round, parameters, client_manager)
        return _fit_round(self.recorder, super().configure_fit, arguments)

    def configure_evaluate(self, server_round, parameters, client_manager):
        arguments = (server_round, parameters, client_manager)
        return _evaluate_round(self.recorder, super().configure_evaluate, arguments)


def build_origin_strategy(job, n_clients, recorder, initial_parameters):
    """FedAvg, or Flower's FedProx for `kind='fedprox'`, with every client every round."""
    options = {
        "fraction_fit": 1.0,
        "fraction_evaluate": 1.0,
        "min_fit_clients": n_clients,
        "min_evaluate_clients": n_clients,
        "min_available_clients": n_clients,
        "initial_parameters": fl.common.ndarrays_to_parameters(initial_parameters),
        "on_fit_config_fn": lambda server_round: {"server_round": server_round},
        "fit_metrics_aggregation_fn": recorder.on_fit,
        "evaluate_metrics_aggregation_fn": recorder.on_evaluate,
    }
    if job["kind"] == "fedprox":
        strategy = OriginFedProx(proximal_mu=job["mu"], **options)
    else:
        strategy = OriginFedAvg(**options)
    strategy.recorder = recorder
    return strategy


def main():
    """Run a Flower server for standalone state clients."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="0.0.0.0:8080")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--min-clients", type=int, required=True)
    args = parser.parse_args()
    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=build_strategy(args.min_clients),
    )


if __name__ == "__main__":
    main()
