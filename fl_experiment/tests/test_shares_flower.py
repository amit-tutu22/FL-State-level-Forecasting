"""Flower share clients reproduce the in-process FH-PFL v2 loop."""
# pylint: disable=missing-function-docstring

import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np
import ray

from fl_experiment.shares import forecast_shares, run_share_origin, share_clients
from fl_experiment.shares_flower import flower_share_origin
from fl_experiment.tests.test_shares import SETTINGS, share_group


class FlowerShareTest(unittest.TestCase):
    """Dropout is 0 in SETTINGS, so both paths use no random numbers after initialization."""

    @classmethod
    def setUpClass(cls):
        logging.getLogger("flwr").setLevel(logging.WARNING)

    @classmethod
    def tearDownClass(cls):
        if ray.is_initialized():
            ray.shutdown()

    def test_flower_matches_in_process_and_uploads_only_month_scalars(self):
        group, folds = share_group()
        clients = share_clients(group, folds)
        best = run_share_origin(clients, SETTINGS, coupled=True)
        expected = forecast_shares(best["origin_scores"], True)
        with tempfile.TemporaryDirectory() as directory:
            shares, info = flower_share_origin(clients, SETTINGS, Path(directory) / "cache")
        found = np.asarray([shares[client["state_id"]] for client in clients])
        np.testing.assert_allclose(found, expected, rtol=1e-5)
        self.assertEqual(info["selected_round"], best["round"])
        allowed = {(len(clients[0]["fit_months"]),), (len(clients[0]["early_months"]),), (1,)}
        self.assertTrue(info["upload_shapes"])
        self.assertTrue(set(info["upload_shapes"]) <= allowed)


if __name__ == "__main__":
    unittest.main()
