"""Flower clients reproduce the in-process FedAvg, FedProx and APFL updates."""
# pylint: disable=missing-function-docstring

import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np
import ray
from flwr.common import Context, RecordSet

from fl_experiment.client import OriginClient
from fl_experiment.federated import (
    apfl_origin,
    fedavg_origin,
    initial_parameters,
    make_job,
)
from fl_experiment.flower_rolling import flower_origin
from fl_experiment.tests.test_matched import SETTINGS, _clients


def _node():
    return {"context": Context(0, {}, RecordSet(), {}), "partition": 0}


def _predictions(rows):
    return np.asarray([row["predicted"] for row in rows])


class FlowerOriginTest(unittest.TestCase):
    """Dropout is 0 in SETTINGS, so training uses no random numbers after initialization."""

    @classmethod
    def setUpClass(cls):
        logging.getLogger("flwr").setLevel(logging.WARNING)

    @classmethod
    def tearDownClass(cls):
        if ray.is_initialized():
            ray.shutdown()

    def _flower(self, job):
        clients, _ = _clients()
        with tempfile.TemporaryDirectory() as directory:
            rows, _ = flower_origin(clients, SETTINGS, job, Path(directory) / "cache")
        return _predictions(rows)

    def test_flower_fedavg_and_fedprox_mu_zero_match_in_process(self):
        clients, _ = _clients()
        expected = _predictions(fedavg_origin(clients, SETTINGS)[0])
        np.testing.assert_allclose(self._flower(make_job("fedavg")), expected, rtol=1e-5)
        np.testing.assert_allclose(self._flower(make_job("fedprox", 0.0)), expected, rtol=1e-5)

    def test_flower_apfl_matches_in_process(self):
        clients, _ = _clients()
        expected = _predictions(apfl_origin(clients, SETTINGS, alpha_step=1.0)[0])
        np.testing.assert_allclose(self._flower(make_job("apfl", 1.0)), expected, rtol=1e-5)

    def test_client_applies_the_proximal_term(self):
        clients, _ = _clients()
        width = clients[0]["fit"][0].shape[-1]
        start = initial_parameters(width, SETTINGS)
        settings = dict(SETTINGS, local_epochs=5)
        client = OriginClient(clients[0], settings, make_job("fedprox", 0.0), _node())
        plain, _, _ = client.fit(start, {"server_round": 1, "proximal_mu": 0.0})
        proximal, _, _ = client.fit(start, {"server_round": 1, "proximal_mu": 10.0})
        moved_plain = sum(float(np.sum((a - b) ** 2)) for a, b in zip(plain, start))
        moved_proximal = sum(float(np.sum((a - b) ** 2)) for a, b in zip(proximal, start))
        self.assertLess(moved_proximal, moved_plain)

    def test_apfl_alpha_is_kept_in_the_node_state(self):
        clients, _ = _clients()
        width = clients[0]["fit"][0].shape[-1]
        node = _node()
        client = OriginClient(clients[0], SETTINGS, make_job("apfl", 1000.0), node)
        _, _, metrics = client.fit(initial_parameters(width, SETTINGS), {"server_round": 1})
        stored = float(node["context"].state.configs_records["apfl"]["alpha"])
        self.assertEqual(stored, metrics["alpha"])
        self.assertTrue(0.0 <= stored <= 1.0)


if __name__ == "__main__":
    unittest.main()
