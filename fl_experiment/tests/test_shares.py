"""FH-PFL v2: gradient identity, coherence, uploads, leakage and the A1 reduction."""
# pylint: disable=missing-function-docstring,too-many-locals

import copy
import unittest

import numpy as np
import torch

from fl_experiment.baselines import GRIDS
from fl_experiment.federated import initial_parameters, load_parameters, model_parameters
from fl_experiment.folds import build_folds
from fl_experiment.model import RevenueModel
from fl_experiment.rolling import states_for_month
from fl_experiment.shares import (
    MODEL_NAMES,
    coupling,
    decode_scores,
    error_upload,
    federation_months,
    joint_objective,
    local_step,
    national_anchors,
    run_share_origin,
    same_federation,
    score_upload,
    share_clients,
    surrogate,
    top_down_rows,
    trained_rows,
    zero_broadcast,
)
from fl_experiment.tests.test_matched import SETTINGS as MATCHED
from fl_experiment.tests.test_state_clients import monthly_records

SETTINGS = dict(MATCHED, local_epochs=2, max_rounds=3)
MONTH = "2021-01"
GAP = "2017-03"


def share_folds():
    """Three states of different size. State B misses one early month."""
    records = (
        monthly_records("A", "2016-01", 84)
        + [row for row in monthly_records("B", "2016-01", 84, base=50) if row[1] != GAP]
        + monthly_records("C", "2016-01", 84, base=20)
    )
    return build_folds(records, ["gst", "other"], "2021-01", "2022-01")


def share_group(folds=None, month=MONTH):
    folds = folds or share_folds()
    return states_for_month(folds, month, "val"), folds


def _model(parameters, width):
    model = RevenueModel(width, SETTINGS["hidden"], SETTINGS["dropout"])
    load_parameters(model, parameters)
    return model


def _distinct_parameters(clients):
    width = int(clients[0]["fit"].shape[-1])
    return [initial_parameters(width, dict(SETTINGS, seed=index))
            for index in range(len(clients))]


class FederationTests(unittest.TestCase):
    """N_t is defined on the months every state shares."""

    def test_federation_months_skip_any_state_gap(self):
        group, folds = share_group()
        self.assertEqual({state_id for state_id, _, _ in group}, {"A", "B", "C"})
        self.assertNotIn(GAP, federation_months(group))
        clients = share_clients(group, folds)
        for client in clients:
            self.assertEqual(client["fit_months"], clients[0]["fit_months"])
            self.assertTrue(all(month > "2017-08" or month < GAP
                                for month in client["fit_months"]))
            self.assertEqual(len(client["early_months"]), 6)
            self.assertLess(max(client["early_months"]), MONTH)

    def test_a_changing_federation_stops_the_run(self):
        group, _ = share_group()
        with self.assertRaises(ValueError):
            same_federation({"2021-01": group, "2021-02": group[:2]})

    def test_national_input_column_is_optional(self):
        group, folds = share_group()
        with_column = share_clients(group, folds, national_input=True)
        without = share_clients(group, folds, national_input=False)
        self.assertEqual(with_column[0]["fit"].shape[-1], without[0]["fit"].shape[-1] + 1)


class ObjectiveTests(unittest.TestCase):
    """The local surrogate is a scaled gradient of the joint objective."""

    def test_surrogate_gradient_equals_the_joint_gradient(self):
        group, folds = share_group()
        clients = share_clients(group, folds)
        parameters = _distinct_parameters(clients)
        width = int(clients[0]["fit"].shape[-1])
        broadcast = coupling(parameters, clients, SETTINGS)
        offset = torch.as_tensor(broadcast["offset"])
        total = torch.as_tensor(broadcast["total_error"])
        models = [_model(params, width) for params in parameters]
        scores = [decode_scores(model(torch.from_numpy(client["fit"])), client)
                  for model, client in zip(models, clients)]
        shares = [torch.from_numpy(client["log_share"]) for client in clients]
        joint_objective(scores, shares).backward()
        joint = [[param.grad.clone() for param in model.parameters()] for model in models]
        for index, client in enumerate(clients):
            model = _model(parameters[index], width)
            local = decode_scores(model(torch.from_numpy(client["fit"])), client)
            surrogate(local, shares[index], offset, total).backward()
            for param, expected in zip(model.parameters(), joint[index]):
                np.testing.assert_allclose(
                    param.grad.numpy(), len(clients) * expected.numpy(), rtol=0, atol=1e-6
                )

    def test_uncoupled_step_is_plain_log_share_regression(self):
        group, folds = share_group()
        client = share_clients(group, folds)[0]
        start = _distinct_parameters([client])[0]
        stepped = local_step(start, client, zero_broadcast(client), SETTINGS)
        model = _model(start, int(client["fit"].shape[-1]))
        optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS["lr"])
        target = torch.from_numpy(client["log_share"])
        for _ in range(SETTINGS["local_epochs"]):
            optimizer.zero_grad()
            scores = decode_scores(model(torch.from_numpy(client["fit"])), client)
            torch.mean((scores - target) ** 2).backward()
            optimizer.step()
        for left, right in zip(stepped, model_parameters(model)):
            np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)


class ForecastTests(unittest.TestCase):
    """Coherence, uploads and leakage."""

    def test_v2_forecasts_add_up_to_each_anchor(self):
        group, folds = share_group()
        clients = share_clients(group, folds)
        best = run_share_origin(clients, SETTINGS, coupled=True)
        anchor = {"sarima": 1234.5, "fallback": False, "naive": 1000.0, "actual": 1500.0}
        rows = trained_rows("v2", clients, best["origin_scores"], anchor)
        totals = {}
        for row in rows:
            totals[row["model"]] = totals.get(row["model"], 0.0) + row["predicted"]
        self.assertAlmostEqual(totals[MODEL_NAMES["v2"]], 1234.5, places=6)
        self.assertAlmostEqual(totals[MODEL_NAMES["a4"]], 1000.0, places=6)
        self.assertAlmostEqual(totals[MODEL_NAMES["oracle"]], 1500.0, places=6)
        independent = trained_rows("a1", clients, best["origin_scores"], anchor)
        posthoc = [row for row in independent if row["model"] == MODEL_NAMES["a2"]]
        self.assertAlmostEqual(sum(row["predicted"] for row in posthoc), 1234.5, places=6)
        self.assertEqual(len(top_down_rows(clients, anchor)), len(clients))

    def test_uploads_are_one_scalar_per_fit_month(self):
        group, folds = share_group()
        clients = share_clients(group, folds)
        parameters = _distinct_parameters(clients)
        months = len(clients[0]["fit_months"])
        offset = np.zeros(months)
        for params, client in zip(parameters, clients):
            self.assertEqual(score_upload(params, client, SETTINGS).shape, (months,))
            self.assertEqual(error_upload(params, client, offset, SETTINGS).shape, (months,))

    def test_origin_and_later_targets_are_not_read(self):
        group, folds = share_group()
        before = share_clients(group, folds)
        anchors, _ = national_anchors({MONTH: group}, "2021-01", GRIDS["small"], 24, 50)
        result = run_share_origin(before, SETTINGS, coupled=True)
        changed = copy.deepcopy(folds)
        for state in changed["states"].values():
            for index, month in enumerate(state["months"]):
                if month >= MONTH:
                    state["targets"][index] = 99999.0
        group_after, _ = share_group(changed)
        after = share_clients(group_after, changed)
        for left, right in zip(before, after):
            for key in ("fit", "log_share", "early", "early_actual", "early_national", "origin"):
                np.testing.assert_array_equal(left[key], right[key])
        again, _ = national_anchors({MONTH: group_after}, "2021-01", GRIDS["small"], 24, 50)
        self.assertEqual(anchors[MONTH]["sarima"], again[MONTH]["sarima"])
        self.assertEqual(anchors[MONTH]["naive"], again[MONTH]["naive"])
        np.testing.assert_array_equal(
            result["origin_scores"],
            run_share_origin(after, SETTINGS, coupled=True)["origin_scores"],
        )


if __name__ == "__main__":
    unittest.main()
