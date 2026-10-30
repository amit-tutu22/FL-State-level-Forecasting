"""FH-PFL v3: SARIMA reduction, leakage, decoding, shrinkage and the federated path."""
# pylint: disable=missing-function-docstring

import copy
import unittest

import numpy as np

from fl_experiment.baselines import GRIDS, forecast_sarima
from fl_experiment.federated import decode_crore, make_job, run_origin
from fl_experiment.folds import build_folds
from fl_experiment.residual import (
    LAMBDAS,
    choose_lambda,
    federated_residuals,
    filtered_base,
    frozen_orders,
    residual_client,
    residual_columns,
    ridge_residuals,
    sarima_bases,
    shrunk_rows,
)
from fl_experiment.rolling import states_for_month
from fl_experiment.state_data import shift_month
from fl_experiment.tests.test_shares import SETTINGS

NAMES = ["feature_gst", "feature_ewb", "feature_cpi_index"]
MONTH = "2021-01"
MAXITER = 50


def seasonal_records(state, base, seed, count=84):
    """Growing, seasonal, noisy GST with an E-Way-Bill-like and a CPI input."""
    rng = np.random.default_rng(seed)
    records, previous = [], base
    for offset in range(count):
        month = shift_month("2016-01", offset)
        level = base * (1 + 0.01 * offset) * (1 + 0.2 * np.sin(2 * np.pi * offset / 12))
        value = float(level * np.exp(0.05 * rng.standard_normal()))
        ewb = float(value * np.exp(0.05 * rng.standard_normal()))
        records.append((state, month, [previous, ewb, 100.0 + offset], value))
        previous = value
    return records


def residual_folds(records=None):
    records = records or (
        seasonal_records("A", 1000.0, 0)
        + seasonal_records("B", 300.0, 1)
        + seasonal_records("C", 50.0, 2)
    )
    return build_folds(records, NAMES, "2021-01", "2022-01")


def prepared(folds=None, month=MONTH, inputs="gst_ewb"):
    folds = folds or residual_folds()
    group = states_for_month(folds, month, "val")
    protocol = folds["protocol"]
    orders = frozen_orders({month: group}, protocol, GRIDS["small"], MAXITER)
    bases = sarima_bases(group, orders, protocol, MAXITER)
    columns = residual_columns(folds["feature_names"], inputs)
    clients = [residual_client(item, bases[item[0]], columns, protocol) for item in group]
    return {"folds": folds, "group": group, "orders": orders, "bases": bases, "clients": clients}


class ColumnTests(unittest.TestCase):
    """CPI never enters the residual GRU."""

    def test_columns_drop_cpi_and_keep_the_month_encoding(self):
        self.assertEqual(residual_columns(NAMES, "gst_ewb"), [0, 1, 3, 4])
        self.assertEqual(residual_columns(NAMES, "gst_only"), [0, 3, 4])
        with self.assertRaises(ValueError):
            residual_columns(NAMES, "all")


class SarimaBaseTests(unittest.TestCase):
    """The two SARIMA fits read nothing at or after the origin month."""

    def setUp(self):
        self.ready = prepared()

    def test_filtered_base_stops_before_the_origin_and_covers_the_holdout(self):
        holdout = self.ready["folds"]["protocol"]["holdout_months"]
        for state_id, base in self.ready["bases"].items():
            months = sorted(base["filtered"])
            self.assertLess(months[-1], MONTH)
            self.assertEqual(months[-1], shift_month(MONTH, -1))
            self.assertEqual(months[0], shift_month("2016-01", 13))
            self.assertTrue(all(value > 0 for value in base["filtered"].values()), state_id)
            self.assertGreaterEqual(len(months), 24 + holdout)

    def test_forecast_fit_is_the_sarima_baseline(self):
        protocol = self.ready["folds"]["protocol"]
        for state_id, state, _ in self.ready["group"]:
            expected, _ = forecast_sarima(
                state, MONTH, self.ready["orders"][state_id], protocol["min_train"], MAXITER
            )
            self.assertEqual(self.ready["bases"][state_id]["forecast"], expected)

    def test_holdout_values_do_not_move_the_fit_month_predictions(self):
        state = self.ready["folds"]["states"]["A"]
        changed = copy.deepcopy(state)
        index = changed["months"].index(shift_month(MONTH, -2))
        changed["targets"][index] *= 3.0
        protocol = self.ready["folds"]["protocol"]
        chosen = self.ready["orders"]["A"]
        before = filtered_base(state, MONTH, chosen, protocol, MAXITER)
        after = filtered_base(changed, MONTH, chosen, protocol, MAXITER)
        fit_end = shift_month(MONTH, -protocol["holdout_months"])
        for month, value in before.items():
            if month <= shift_month(MONTH, -2):
                self.assertAlmostEqual(after[month], value, places=6, msg=month)
        self.assertNotAlmostEqual(after[shift_month(MONTH, -1)], before[shift_month(MONTH, -1)])
        self.assertIn(fit_end, before)

    def test_changing_the_origin_and_later_months_changes_no_input(self):
        records = (
            seasonal_records("A", 1000.0, 0)
            + seasonal_records("B", 300.0, 1)
            + seasonal_records("C", 50.0, 2)
        )
        later = [
            (state, month, features, value * 5.0 if month >= MONTH else value)
            for state, month, features, value in records
        ]
        first, second = prepared(), prepared(residual_folds(later))
        for left, right in zip(first["clients"], second["clients"]):
            for key in ("fit", "early_stop"):
                np.testing.assert_array_equal(left[key][0], right[key][0])
                np.testing.assert_array_equal(left[key][1], right[key][1])
            np.testing.assert_array_equal(left["origin"][0], right["origin"][0])
            np.testing.assert_array_equal(left["base_origin"], right["base_origin"])
            self.assertNotEqual(left["actual"], right["actual"])


class DecodeTests(unittest.TestCase):
    """Residual targets decode back to the observed GST, and lambda = 0 is SARIMA."""

    def setUp(self):
        self.ready = prepared()

    def test_decode_inverts_the_residual(self):
        for client, (_, state, _) in zip(self.ready["clients"], self.ready["group"]):
            targets = dict(zip(state["months"], state["targets"]))
            fit_end = shift_month(MONTH, -self.ready["folds"]["protocol"]["holdout_months"])
            early = [shift_month(fit_end, offset) for offset in range(6)]
            decoded = decode_crore(
                client["early_stop"][1], client["y_mean"], client["y_std"], client["base_early"]
            )
            np.testing.assert_allclose(decoded, [targets[month] for month in early], rtol=1e-5)

    def test_lambda_zero_reproduces_sarima_exactly(self):
        clients = self.ready["clients"]
        residuals = [ridge_residuals(client) for client in clients]
        rows, lambdas = shrunk_rows("v3", clients, residuals, fixed=0.0)
        for row, client in zip(rows, clients):
            self.assertEqual(row["predicted"], self.ready["bases"][client["state_id"]]["forecast"])
        self.assertEqual(set(lambdas.values()), {0.0})

    def test_lambda_picks_one_for_the_truth_and_zero_for_its_negative(self):
        client = self.ready["clients"][0]
        truth = client["early_stop"][1].reshape(-1) * client["y_std"] + client["y_mean"]
        self.assertEqual(choose_lambda(truth, client), 1.0)
        self.assertEqual(choose_lambda(-truth, client), 0.0)
        self.assertEqual(choose_lambda(np.zeros_like(truth), client), 0.0)
        self.assertEqual(LAMBDAS, (0.0, 0.5, 1.0))


class FederatedTests(unittest.TestCase):
    """The shared FedAvg/APFL loop decodes residual clients through the SARIMA base."""

    def test_full_correction_rows_match_the_shared_loop(self):
        clients = prepared()["clients"]
        job = make_job("apfl", 1.0, 0.5)
        expected, _ = run_origin(clients, SETTINGS, job)
        residuals, info = federated_residuals(clients, SETTINGS, job)
        rows, _ = shrunk_rows("b5", clients, residuals, fixed=1.0)
        np.testing.assert_allclose(
            [row["predicted"] for row in rows], [row["predicted"] for row in expected],
            rtol=1e-6,
        )
        self.assertGreaterEqual(info["selected_round"], 1)

    def test_gst_only_clients_have_one_input_fewer(self):
        wide = prepared()["clients"][0]["fit"][0].shape[-1]
        narrow = prepared(inputs="gst_only")["clients"][0]["fit"][0].shape[-1]
        self.assertEqual(wide, 5)
        self.assertEqual(narrow, 4)


if __name__ == "__main__":
    unittest.main()
