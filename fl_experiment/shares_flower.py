"""FH-PFL v2 through Flower. Clients upload per-month scalars, never weights.

One FH-PFL round is three Flower rounds:

    score  upload exp(g_t) on the fit months, and report the previous round's
           holdout RMSE and origin share from the sums broadcast after it
    error  receive c_t = log Z_t, upload e_t on the fit months
    step   receive c_t and E_t, run the local step, upload exp(g) on the
           holdout months and at the origin month

The strategy sums each upload across clients and broadcasts only the sums.
The sums stand in for secure aggregation; nothing here is encrypted. Each
state's weights stay in its node state. A last score call closes the final
round. The update maths is the in-process code in `shares.py`.
"""

import json
from collections import OrderedDict
from pathlib import Path

import flwr as fl
import numpy as np
import torch
from flwr.common import (
    ConfigsRecord,
    Context,
    FitIns,
    ParametersRecord,
    array_from_numpy,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.simulation import start_simulation

from fl_experiment.federated import initial_parameters
from fl_experiment.server import OriginRecorder
from fl_experiment.shares import (
    error_upload,
    holdout_rmse,
    local_step,
    log_scores,
    score_upload,
    secure_sum,
)

ARRAYS = ("fit", "log_share", "early", "early_actual", "early_national", "origin")
META = ("state_id", "target_month", "actual", "naive", "share_mean", "share_std")
RAY_ARGS = {"include_dashboard": False, "ignore_reinit_error": True, "log_to_driver": False}


def write_share_cache(clients, directory, settings):
    """One file per state, so each Ray worker loads only its own state's arrays."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for index, client in enumerate(clients):
        header = {"meta": {key: client[key] for key in META}, "settings": settings}
        np.savez(
            directory / f"share_{index}.npz",
            header=np.asarray(json.dumps(header)),
            **{key: client[key] for key in ARRAYS},
        )
    return directory


def load_share_cache(directory, partition_id):
    """Return (client, settings) for one partition of a share cache."""
    with np.load(Path(directory) / f"share_{partition_id}.npz") as stored:
        header = json.loads(str(stored["header"]))
        client = dict(header["meta"], **{key: stored[key] for key in ARRAYS})
    return client, header["settings"]


class ShareClient(fl.client.NumPyClient):
    """One state at one origin. Its GRU weights never leave the node state."""

    def __init__(self, client, settings, node):
        torch.set_num_threads(1)
        self.client = client
        self.settings = settings
        self.node = node

    def _state(self):
        return self.node["context"].state

    def _parameters(self):
        record = self._state().parameters_records.get("shares")
        tag = self._state().configs_records.get("shares")
        if record is None or tag is None or tag["key"] != self.node["key"]:
            width = int(self.client["fit"].shape[-1])
            return initial_parameters(width, self.settings)
        return [record[str(index)].numpy() for index in range(len(record))]

    def _store(self, parameters):
        arrays = OrderedDict(
            (str(index), array_from_numpy(np.ascontiguousarray(layer)))
            for index, layer in enumerate(parameters)
        )
        self._state().parameters_records["shares"] = ParametersRecord(arrays)
        self._state().configs_records["shares"] = ConfigsRecord({"key": self.node["key"]})

    def _close(self, parameters, sums):
        """Score the previous round from the broadcast Z on the holdout and at the origin."""
        early = log_scores(parameters, self.client, "early", self.settings)
        origin = log_scores(parameters, self.client, "origin", self.settings)[0]
        return {
            "state_id": self.client["state_id"],
            "holdout_rmse": holdout_rmse(early, sums[0], self.client),
            "prediction": float(np.exp(origin) / float(sums[1][0])),
        }

    def _step(self, parameters, broadcast, server_round):
        torch.manual_seed(
            self.settings["seed"] * 1_000_003 + server_round * 1_009 + self.node["partition"]
        )
        parameters = local_step(
            parameters, self.client,
            {"offset": broadcast[0], "total_error": broadcast[1]}, self.settings,
        )
        self._store(parameters)
        return [
            np.exp(log_scores(parameters, self.client, "early", self.settings)),
            np.exp(log_scores(parameters, self.client, "origin", self.settings)),
        ]

    def fit(self, parameters, config):
        phase = config["phase"]
        current = self._parameters()
        count = len(self.client["fit"])
        if phase == "score":
            metrics = self._close(current, parameters) if parameters else {}
            return [score_upload(current, self.client, self.settings)], count, metrics
        if phase == "error":
            return [error_upload(current, self.client, parameters[0], self.settings)], count, {}
        return self._step(current, parameters, int(config["server_round"])), count, {}

    def evaluate(self, parameters, config):
        return 0.0, 0, {}


def make_share_client_fn(cache):
    """Flower `client_fn` for one share cache."""
    cache = str(cache)

    def client_fn(context: Context):
        partition = int(context.node_config["partition-id"])
        client, settings = load_share_cache(cache, partition)
        node = {"context": context, "partition": partition, "key": cache}
        return ShareClient(client, settings, node).to_client()

    return client_fn


class ShareStrategy(fl.server.strategy.Strategy):
    """Sums per-month uploads across every client and broadcasts the sums."""

    def __init__(self, n_clients, recorder, max_rounds):
        super().__init__()
        self.n_clients = n_clients
        self.recorder = recorder
        self.max_rounds = max_rounds
        self.phase = "score"
        self.share_round = 1
        self.broadcast = []
        self.upload_shapes = set()

    def initialize_parameters(self, client_manager):
        return ndarrays_to_parameters([])

    def configure_fit(self, server_round, parameters, client_manager):
        if self.recorder.stopped:
            return []
        clients = client_manager.sample(
            num_clients=self.n_clients, min_num_clients=self.n_clients
        )
        instructions = FitIns(
            ndarrays_to_parameters(self.broadcast),
            {"phase": self.phase, "server_round": server_round},
        )
        return [(client, instructions) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        if failures:
            raise RuntimeError(f"{len(failures)} share clients failed")
        uploads = [parameters_to_ndarrays(result.parameters) for _, result in results]
        for upload in uploads:
            self.upload_shapes.update(tuple(item.shape) for item in upload)
        sums = [secure_sum([upload[index] for upload in uploads])
                for index in range(len(uploads[0]))]
        if self.phase == "score":
            self._after_score(sums, [(result.num_examples, result.metrics)
                                     for _, result in results])
        elif self.phase == "error":
            self.broadcast = [self.broadcast[0], sums[0]]
            self.phase = "step"
        else:
            self.broadcast = sums
            self.phase = "score"
            self.share_round += 1
        return None, {}

    def _after_score(self, sums, results):
        if self.share_round > 1:
            self.recorder.current_round = self.share_round - 1
            self.recorder.close_round(results)
        if self.share_round > self.max_rounds:
            self.recorder.stopped = True
        if self.recorder.stopped:
            return
        self.broadcast = [np.log(sums[0])]
        self.phase = "error"

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round, parameters):
        return None


def flower_share_origin(clients, settings, cache):
    """One Flower simulation of FH-PFL v2 at one origin.

    Returns the best round's normalised origin shares by state, plus run info.
    """
    write_share_cache(clients, cache, settings)
    recorder = OriginRecorder(settings["round_patience"], personalize=False)
    strategy = ShareStrategy(len(clients), recorder, settings["max_rounds"])
    start_simulation(
        client_fn=make_share_client_fn(cache),
        num_clients=len(clients),
        config=fl.server.ServerConfig(num_rounds=3 * settings["max_rounds"] + 1),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args=RAY_ARGS,
        keep_initialised=True,
    )
    info = {
        "selected_round": recorder.best["round"],
        "rounds_run": len(recorder.history),
        "upload_shapes": sorted(strategy.upload_shapes),
    }
    return recorder.best["predictions"], info
