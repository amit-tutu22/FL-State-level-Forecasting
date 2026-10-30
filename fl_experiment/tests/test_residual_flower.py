"""Flower clients reproduce the in-process FH-PFL v3 full-correction model."""
# pylint: disable=missing-function-docstring

import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np
import ray

from fl_experiment.federated import make_job, run_origin
from fl_experiment.flower_rolling import flower_origin
from fl_experiment.tests.test_residual import prepared
from fl_experiment.tests.test_shares import SETTINGS


class FlowerResidualTest(unittest.TestCase):
    """Dropout is 0 in SETTINGS, so both paths use no random numbers after initialization."""

    @classmethod
    def setUpClass(cls):
        logging.getLogger("flwr").setLevel(logging.WARNING)

    @classmethod
    def tearDownClass(cls):
        if ray.is_initialized():
            ray.shutdown()

    def test_flower_apfl_on_residual_clients_matches_in_process(self):
        clients = prepared()["clients"]
        job = make_job("apfl", 1.0, 0.5)
        expected, info = run_origin(clients, SETTINGS, job)
        with tempfile.TemporaryDirectory() as directory:
            rows, flower_info = flower_origin(clients, SETTINGS, job, Path(directory) / "cache")
        np.testing.assert_allclose(
            [row["predicted"] for row in rows], [row["predicted"] for row in expected],
            rtol=1e-5,
        )
        self.assertEqual(flower_info["selected_round"], info["selected_round"])


if __name__ == "__main__":
    unittest.main()
