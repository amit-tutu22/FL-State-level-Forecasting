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
