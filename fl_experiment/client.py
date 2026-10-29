"""Flower client for one state, usable in simulation or as a standalone process.

Standalone (one process per state, after `run_fl_simulation --prepare-only`):
    python -m fl_experiment.client --cache DIR --partition-id 0 --server 127.0.0.1:8080
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import flwr as fl
import numpy as np
import torch
from flwr.common import ConfigsRecord, Context

from fl_experiment.federated import (
    client_holdout_rmse,
    client_origin_prediction,
    client_step,
)
from fl_experiment.metrics import regression_metrics
from fl_experiment.model import RevenueModel
from fl_experiment.state_data import load_cached_client

BLOCKS = ("fit", "early_stop", "origin")


class StateClient(fl.client.NumPyClient):
    """Trains on its state's training period and evaluates on one held-out period."""

    def __init__(self, data, local_epochs=5, lr=0.01, eval_split="val", seed=0):
        if eval_split not in {"val", "test"}:
            raise ValueError("eval_split must be 'val' or 'test'")
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        self.data = data
        self.local_epochs = local_epochs
        self.lr = lr
        self.eval_split = eval_split
        self.model = RevenueModel(data.n_features)
        self.loss_fn = torch.nn.MSELoss()

    def get_parameters(self, config):
        return [value.detach().cpu().numpy() for value in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        """Load global weights in `state_dict` order."""
        keys = self.model.state_dict().keys()
        state = OrderedDict((k, torch.tensor(np.asarray(v))) for k, v in zip(keys, parameters))
        self.model.load_state_dict(state, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        x = torch.from_numpy(self.data.x["train"])
        y = torch.from_numpy(self.data.y["train"])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.model.train()
        loss = torch.zeros(())
        for _ in range(self.local_epochs):
            optimizer.zero_grad()
            loss = self.loss_fn(self.model(x), y)
            loss.backward()
            optimizer.step()
        return self.get_parameters(config), len(x), {"train_loss": float(loss.item())}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        x = torch.from_numpy(self.data.x[self.eval_split])
        y = torch.from_numpy(self.data.y[self.eval_split])
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(x)
            loss = float(self.loss_fn(prediction, y).item())
        metrics = regression_metrics(
            self.data.decode(y.numpy()), self.data.decode(prediction.numpy())
        )
        metrics["state_id"] = self.data.state_id
        return loss, len(x), metrics


def write_origin_cache(clients, directory, settings, job):
    """Write one origin's per-state arrays, so Ray workers can load their own state."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    states = []
    for index, client in enumerate(clients):
        arrays = {}
        for block in BLOCKS:
            arrays[f"{block}_x"], arrays[f"{block}_y"] = client[block]
        np.savez(directory / f"client_{index}.npz", **arrays)
        states.append({
            key: client[key]
            for key in ("state_id", "target_month", "actual", "naive", "y_mean", "y_std")
        })
    manifest = {"states": states, "settings": settings, "job": job}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def load_origin_cache(directory, partition_id):
    """Return (client, settings, job) for one partition of an origin cache."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    client = dict(manifest["states"][partition_id])
    with np.load(directory / f"client_{partition_id}.npz") as arrays:
        for block in BLOCKS:
            client[block] = (arrays[f"{block}_x"], arrays[f"{block}_y"])
    return client, manifest["settings"], manifest["job"]


class OriginClient(fl.client.NumPyClient):
    """One state at one rolling origin. The update maths lives in `federated.py`.

    FedProx reads `proximal_mu` from the fit config. APFL keeps its mixing
    weight in the node's context state, so it carries over between rounds.
    """

    def __init__(self, client, settings, job, node):
        torch.set_num_threads(1)
        self.client = client
        self.settings = settings
        self.job = job
        self.node = node
        self.n_features = int(client["fit"][0].shape[-1])

    def _alpha(self):
        record = self.node["context"].state.configs_records.get("apfl")
        return float(record["alpha"]) if record else self.job["alpha_init"]

    def _scores(self, parameters):
        return {
            "state_id": self.client["state_id"],
            "holdout_rmse": client_holdout_rmse(
                parameters, self.client, self.n_features, self.settings
            ),
            "prediction": client_origin_prediction(
                parameters, self.client, self.n_features, self.settings
            ),
        }

    def fit(self, parameters, config):
        server_round = int(config.get("server_round", 0))
        torch.manual_seed(
            self.settings["seed"] * 1_000_003 + server_round * 1_009 + self.node["partition"]
        )
        job = dict(self.job, mu=float(config.get("proximal_mu", 0.0)))
        local, personal, alpha = client_step(
            parameters, self.client, self._alpha(), self.n_features, self.settings, job
        )
        metrics = {}
        if job["personalize"]:
            self.node["context"].state.configs_records["apfl"] = ConfigsRecord({"alpha": alpha})
            metrics = {**self._scores(personal), "alpha": alpha}
        return local, len(self.client["fit"][0]), metrics

    def evaluate(self, parameters, config):
        scores = self._scores(parameters)
        return scores["holdout_rmse"], len(self.client["early_stop"][0]), scores


def make_origin_client_fn(cache):
    """Flower `client_fn` for one origin cache."""
    cache = str(cache)

    def client_fn(context: Context):
        partition = int(context.node_config["partition-id"])
        client, settings, job = load_origin_cache(cache, partition)
        node = {"context": context, "partition": partition}
        return OriginClient(client, settings, job, node).to_client()

    return client_fn


def make_client_fn(cache, local_epochs=5, lr=0.01, eval_split="val", seed=0):
    """Build a Flower `client_fn` that loads the partition assigned to each node."""
    cache = str(cache)

    def client_fn(context: Context):
        partition_id = int(context.node_config["partition-id"])
        data = load_cached_client(cache, partition_id)
        return StateClient(data, local_epochs, lr, eval_split, seed + partition_id).to_client()

    return client_fn


def main():
    """Connect one state client to a running Flower server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, help="client cache directory")
    parser.add_argument("--partition-id", type=int, required=True)
    parser.add_argument("--server", default="127.0.0.1:8080")
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    data = load_cached_client(args.cache, args.partition_id)
    client = StateClient(data, args.local_epochs, args.lr, "val", args.seed + args.partition_id)
    fl.client.start_client(server_address=args.server, client=client.to_client())


if __name__ == "__main__":
    main()
