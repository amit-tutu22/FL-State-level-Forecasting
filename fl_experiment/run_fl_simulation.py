"""Run a local Flower FedAvg simulation over 10 synthetic state clients.

Usage (from repo root or fl_experiment/):
    python run_fl_simulation.py
    python run_fl_simulation.py --rounds 3 --clients 10
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
from flwr.simulation import start_simulation

from client import client_fn


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total = sum(num for num, _ in metrics)
    if total == 0:
        return {}
    return {
        key: sum(num * m[key] for num, m in metrics if key in m) / total
        for key in metrics[0][1]
    }


def main():
    parser = argparse.ArgumentParser(description="Flower FL simulation (Phase 1 forecast)")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--min-clients", type=int, default=5)
    args = parser.parse_args()

    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=args.min_clients,
        min_available_clients=args.min_clients,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
    )

    print(
        f"Starting FedAvg forecast simulation: {args.clients} clients, "
        f"{args.rounds} rounds, min_clients={args.min_clients}"
    )
    hist = start_simulation(
        client_fn=client_fn,
        num_clients=args.clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )
    print("Simulation finished.")
    if hist.losses_distributed:
        print("Final distributed loss (test MSE):", hist.losses_distributed[-1])
    if hist.metrics_distributed:
        print("Final distributed metrics:", hist.metrics_distributed)
    return hist


if __name__ == "__main__":
    main()
