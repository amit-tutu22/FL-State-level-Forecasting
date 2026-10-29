"""FedProx match, APFL bounds, and the test-period gate."""
# pylint: disable=missing-function-docstring

import sys
import unittest
from unittest import mock

import numpy as np

from fl_experiment.evaluate_test import require_selection
from fl_experiment.federated import (
    apfl_origin,
    clip_alpha,
    fedavg_origin,
    fedprox_origin,
    initial_parameters,
    mix_parameters,
    predict_crore,
    update_alpha,
)
from fl_experiment.folds import build_folds, pooled_scales, protocol_hash, shift_month
from fl_experiment.rolling import federated_clients, states_for_month
from fl_experiment.run_fl_simulation import main as simulation_main
from fl_experiment.tests.test_state_clients import monthly_records

SETTINGS = {
    "epochs": 2,
    "patience": 1,
    "lr": 0.05,
    "weight_decay": 0.0,
    "hidden": 4,
    "dropout": 0.0,
    "seed": 0,
    "local_epochs": 1,
    "max_rounds": 2,
    "round_patience": 2,
}


def _clients():
    records = monthly_records("A", "2018-01", 48) + monthly_records("B", "2018-01", 48, base=80)
    folds = build_folds(records, ["gst", "other"], "2021-01", "2022-01")
    group = states_for_month(folds, "2021-01", "val")
    return federated_clients(group, "all", folds), folds


class MatchedProtocolTest(unittest.TestCase):
    """Sanity checks that do not open the real test year."""
    def test_mu_zero_matches_fedavg(self):
        clients, _ = _clients()
        averaged, _ = fedavg_origin(clients, SETTINGS)
        proximal, _ = fedprox_origin(clients, SETTINGS, mu=0.0)
        for left, right in zip(averaged, proximal):
            self.assertEqual(left["state_id"], right["state_id"])
            self.assertAlmostEqual(left["predicted"], right["predicted"], places=6)

    def test_alpha_stays_inside_the_unit_interval(self):
        self.assertEqual(clip_alpha(-2), 0.0)
        self.assertEqual(clip_alpha(3), 1.0)
        self.assertEqual(update_alpha(0.5, step=1000, gradient=1), 0.0)
        self.assertEqual(update_alpha(0.5, step=1000, gradient=-1), 1.0)
        clients, _ = _clients()
        _, info = apfl_origin(clients, SETTINGS, alpha_step=1000)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in info["alphas"]))

    def test_alpha_zero_reproduces_the_global_model(self):
        clients, _ = _clients()
        width = clients[0]["fit"][0].shape[-1]
        global_params = initial_parameters(width, SETTINGS)
        local = [layer + 1.0 for layer in global_params]
        mixed = mix_parameters(local, global_params, 0.0)
        for left, right in zip(mixed, global_params):
            np.testing.assert_allclose(left, right)
        inputs = clients[0]["origin"][0]
        kwargs = {
            "y_mean": clients[0]["y_mean"],
            "y_std": clients[0]["y_std"],
            "n_features": width,
            "settings": SETTINGS,
        }
        global_prediction = predict_crore(global_params, inputs, **kwargs)
        mixed_prediction = predict_crore(mixed, inputs, **kwargs)
        np.testing.assert_allclose(global_prediction, mixed_prediction)

    def test_pooled_scaler_ignores_the_origin_month(self):
        records = monthly_records("A", "2018-01", 48) + monthly_records("B", "2018-01", 48, base=80)
        folds = build_folds(records, ["gst", "other"], "2021-01", "2022-01")
        states = list(folds["states"].values())
        fit_end = shift_month("2021-01", -6)
        before = pooled_scales(states, fit_end, [0, 1, 2, 3])
        for state in states:
            index = state["months"].index("2021-01")
            state["targets"][index] = 99999.0
        after = pooled_scales(states, fit_end, [0, 1, 2, 3])
        np.testing.assert_allclose(before[0], after[0])
        self.assertAlmostEqual(before[2], after[2])

    def test_test_gate_refuses_a_different_fold_hash(self):
        _, folds = _clients()
        selection = {"frozen": True, "protocol_hash": protocol_hash(folds)}
        require_selection(selection, folds)
        selection["protocol_hash"] = "0" * 64
        with self.assertRaises(SystemExit):
            require_selection(selection, folds)
        with self.assertRaises(SystemExit):
            require_selection({"frozen": False, "protocol_hash": protocol_hash(folds)}, folds)

    def test_fixed_split_runner_refuses_the_test_period(self):
        with mock.patch.object(sys, "argv", ["run_fl_simulation", "--final-test"]):
            with self.assertRaises(SystemExit) as caught:
                simulation_main()
        self.assertIn("evaluate_test", str(caught.exception))

    def test_protocol_hash_changes_when_an_actual_changes(self):
        _, folds = _clients()
        original = protocol_hash(folds)
        folds["states"]["A"]["origins"][0]["actual"] = 12345.0
        self.assertNotEqual(original, protocol_hash(folds))


if __name__ == "__main__":
    unittest.main()
