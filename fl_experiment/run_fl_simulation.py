"""Flower FedAvg simulation with one disjoint client per state.

Smoke test on the legacy synthetic CSV:
    python -m fl_experiment.run_fl_simulation --rounds 3
Real panel (assumed-lag sensitivity mode requires explicit opt-in):
    python -m fl_experiment.run_fl_simulation --source panel \
        --panel data/reports/lagged-2012/panel.csv --allow-assumed-vintages
"""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import flwr as fl
import numpy as np
import torch
from flwr.simulation import start_simulation

from fl_experiment.client import make_client_fn
from fl_experiment.server import MetricsRecorder, build_strategy
from fl_experiment.state_data import add_data_arguments, clients_from_arguments, write_cache

RESULTS = Path(__file__).resolve().parent / "results"


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="evaluate on the reserved test period (only after configurations are frozen)",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="write the client cache for standalone server/client processes and exit",
    )
    return parser.parse_args()


def main():
    """Prepare state clients, run the simulation and save the run record."""
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    clients, excluded, manifest = clients_from_arguments(args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir or RESULTS / f"{args.source}-{stamp}"
    cache = write_cache(clients, output / "cache", manifest)

    print(f"{len(clients)} state clients ({args.source}); excluded: {excluded or 'none'}")
    for client in clients:
        summary = client.summary()
        print(
            f"  state {summary['state_id']}: windows train {summary['train']['windows']}, "
            f"val {summary['val']['windows']}, test {summary['test']['windows']}"
        )
    if args.prepare_only:
        print(f"Client cache written to {cache}")
        return

    eval_split = "test" if args.final_test else "val"
    recorder = MetricsRecorder()
    history = start_simulation(
        client_fn=make_client_fn(cache, args.local_epochs, args.lr, eval_split, args.seed),
        num_clients=len(clients),
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=build_strategy(len(clients), recorder),
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args={"include_dashboard": False},
    )

    record = {
        "created_utc": stamp,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "eval_split": eval_split,
        "data": json.loads((cache / "manifest.json").read_text(encoding="utf-8")),
        "loss_by_round": history.losses_distributed,
        "metrics_by_round": history.metrics_distributed,
        "train_metrics_by_round": history.metrics_distributed_fit,
        "per_state_by_round": recorder.per_state,
    }
    record["data"].pop("months")
    (output / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    final = {key: values[-1][1] for key, values in history.metrics_distributed.items()}
    print(f"Final {eval_split} metrics (INR crore): {json.dumps(final)}")
    print(f"Run record: {output / 'run.json'}")


if __name__ == "__main__":
    main()
