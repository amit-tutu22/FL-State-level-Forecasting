"""Client ownership, chronology, windows, preprocessing and model behaviour."""
# pylint: disable=missing-function-docstring

import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from fl_experiment import local_gru
from fl_experiment.client import StateClient
from fl_experiment.server import MetricsRecorder
from fl_experiment.state_data import (
    PANEL_FEATURES,
    PANEL_TARGET,
    build_state_clients,
    check_disjoint,
    load_cached_client,
    load_panel_records,
    load_synthetic_records,
    select_features,
    shift_month,
    window_targets,
    write_cache,
)

CUTOFFS = ("2021-01", "2022-01")


def monthly_records(state, start, count, base=100.0):
    """Synthetic-shaped records with one feature pair per target month."""
    records = []
    for offset in range(count):
        month = shift_month(start, offset)
        records.append((state, month, [base + offset, base + offset - 12], base + offset + 1))
    return records


def one_state(**kwargs):
    clients, _ = build_state_clients(monthly_records("A", "2018-01", 60), *CUTOFFS, **kwargs)
    return clients[0]


class StateClientTests(unittest.TestCase):
    """Ownership and chronology required by issue #18."""

    def test_synthetic_clients_are_disjoint_states_with_ordered_splits(self):
        records, _ = load_synthetic_records()
        clients, excluded = build_state_clients(records, "2022-01", "2023-01")
        self.assertEqual(len(clients), 10)
        self.assertEqual(excluded, {})
        self.assertEqual(len({c.state_id for c in clients}), len(clients))
        for client in clients:
            self.assertLess(max(client.months["train"]), min(client.months["val"]))
            self.assertLess(max(client.months["val"]), min(client.months["test"]))

    def test_each_target_month_is_owned_once(self):
        records, _ = load_synthetic_records()
        clients, _ = build_state_clients(records, "2022-01", "2023-01", window=6)
        owned = [
            (client.state_id, month)
            for client in clients
            for months in client.months.values()
            for month in months
        ]
        self.assertEqual(len(owned), len(set(owned)))
        self.assertTrue(set(owned) <= {(r[0], r[1]) for r in records})
        self.assertEqual(len(owned), len(records) - 5 * len(clients))

    def test_synthetic_inputs_exclude_target_derived_columns(self):
        _, features = load_synthetic_records()
        self.assertNotIn("sgst_share", features)
        self.assertNotIn("cgst_share", features)

    def test_scaling_uses_training_rows_only(self):
        records = monthly_records("A", "2018-01", 60)
        changed = [
            (s, m, [f * 1000 for f in x], y * 1000) if m >= CUTOFFS[0] else (s, m, x, y)
            for s, m, x, y in records
        ]
        first, _ = build_state_clients(records, *CUTOFFS)
        second, _ = build_state_clients(changed, *CUTOFFS)
        np.testing.assert_allclose(first[0].x_mean, second[0].x_mean)
        self.assertAlmostEqual(first[0].y_mean, second[0].y_mean)

    def test_short_or_unevaluable_states_are_excluded_with_reason(self):
        records = monthly_records("A", "2018-01", 60) + monthly_records("B", "2020-06", 10)
        records += monthly_records("C", "2018-01", 30)
        clients, excluded = build_state_clients(records, *CUTOFFS)
        self.assertEqual([c.state_id for c in clients], ["A"])
        self.assertIn("training windows", excluded["B"])
        self.assertEqual(excluded["C"], "no validation or test windows")

    def test_duplicate_state_month_is_rejected(self):
        records = monthly_records("A", "2018-01", 60)
        with self.assertRaises(ValueError):
            build_state_clients(records + records[:1], *CUTOFFS)

    def test_check_disjoint_rejects_repeated_state(self):
        client = one_state()
        with self.assertRaises(ValueError):
            check_disjoint([client, client])

    def test_decode_inverts_target_transform(self):
        client = one_state()
        expected = [r[3] for r in monthly_records("A", "2018-01", 60) if r[1] >= CUTOFFS[1]]
        np.testing.assert_allclose(client.decode(client.y["test"]), expected, rtol=1e-4)

    def test_cache_round_trip(self):
        client = one_state()
        with tempfile.TemporaryDirectory() as folder:
            write_cache([client], folder, {"source": "test"})
            loaded = load_cached_client(folder, 0)
        self.assertEqual(loaded.state_id, "A")
        self.assertEqual(loaded.months, client.months)
        np.testing.assert_array_equal(loaded.x["val"], client.x["val"])

    def test_panel_loader_requires_opt_in_for_assumed_vintages(self):
        header = ["state_id", "target_month", "availability_mode", "complete_features"]
        header += [PANEL_TARGET] + PANEL_FEATURES
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "panel.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(header)
                writer.writerow(
                    ["27", "2024-05", "assumed_lag_latest_vintage", "True", "100"]
                    + ["1"] * len(PANEL_FEATURES)
                )
            with self.assertRaises(ValueError):
                load_panel_records(path)
            records, features = load_panel_records(path, allow_assumed_vintages=True)
        self.assertEqual(records[0][:2], ("27", "2024-05"))
        self.assertEqual(features, PANEL_FEATURES)


class WindowTests(unittest.TestCase):
    """Sequence windows for the GRU."""

    def test_window_shape(self):
        client = one_state(window=6)
        self.assertEqual(client.x["train"].shape, (31, 6, 4))
        self.assertEqual(client.x["val"].shape, (12, 6, 4))
        self.assertEqual(client.n_features, 4)

    def test_windows_never_cross_a_gap(self):
        months = [shift_month("2020-01", i) for i in range(24) if i != 10]
        targets = window_targets(months, 6)
        self.assertNotIn("2020-11", targets)
        for blocked in range(11, 16):
            self.assertNotIn(shift_month("2020-01", blocked), targets)
        self.assertIn(shift_month("2020-01", 16), targets)

    def test_window_inputs_end_at_the_target_month(self):
        records = monthly_records("A", "2018-01", 60)
        client = one_state(window=3)
        month = client.months["val"][0]
        row = next(r for r in records if r[1] == month)
        latest = client.x["val"][0, -1, 0] * client.x_std[0] + client.x_mean[0]
        self.assertAlmostEqual(float(latest), math.log1p(row[2][0]), places=4)
        self.assertLess(max(client.months["train"]), min(client.months["val"]))

    def test_gst_only_keeps_lagged_gst(self):
        records = [("27", "2024-05", list(range(1, len(PANEL_FEATURES) + 1)), 10.0)]
        selected, names = select_features(records, PANEL_FEATURES, "gst_only")
        self.assertEqual(names, ["feature_gst_excluding_cess_crore"])
        self.assertEqual(selected[0][2], [1])


class ModelTests(unittest.TestCase):
    """GRU in the Flower client and the local baseline."""

    def test_client_fit_and_evaluate(self):
        client = StateClient(one_state(), local_epochs=20, lr=0.05)
        initial = client.get_parameters({})
        _, _, before = client.evaluate(initial, {})
        updated, count, fit_metrics = client.fit(initial, {})
        self.assertEqual(count, 31)
        self.assertEqual([p.shape for p in updated], [p.shape for p in initial])
        self.assertTrue(math.isfinite(fit_metrics["train_loss"]))
        loss, count, metrics = client.evaluate(updated, {})
        self.assertEqual(count, 12)
        self.assertEqual(metrics["state_id"], "A")
        self.assertTrue(math.isfinite(loss))
        self.assertEqual(set(before) - {"state_id"}, {"rmse", "mae", "mape"})

    def test_local_gru_early_stopping_restores_best_epoch(self):
        settings = {"epochs": 300, "patience": 5, "lr": 0.05, "weight_decay": 0.0,
                    "hidden": 8, "dropout": 0.0, "seed": 0}
        _, training = local_gru.train_state(one_state(), settings)
        self.assertLessEqual(training["best_epoch"], training["epochs_run"])
        self.assertLess(training["epochs_run"], 300)
        self.assertEqual(training["epochs_run"] - training["best_epoch"], 5)

    def test_local_gru_exports_train_and_validation_only(self):
        with tempfile.TemporaryDirectory() as folder:
            argv = ["local_gru", "--epochs", "20", "--patience", "5", "--output-dir", folder]
            with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"):
                local_gru.main()
            with (Path(folder) / "predictions.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            record = json.loads((Path(folder) / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual({r["split"] for r in rows}, {"train", "val"})
        self.assertEqual(len({r["state_id"] for r in rows}), 10)
        self.assertEqual(record["eval_split"], "val")
        self.assertIn("rmse_equal_state", record["validation_aggregate"])

    def test_recorder_reports_weighted_and_equal_state_means(self):
        recorder = MetricsRecorder()
        result = recorder([(1, {"rmse": 10.0, "mae": 1.0, "mape": 1.0, "state_id": "A"}),
                           (3, {"rmse": 20.0, "mae": 1.0, "mape": 1.0, "state_id": "B"})])
        self.assertAlmostEqual(result["rmse"], 17.5)
        self.assertAlmostEqual(result["rmse_equal_state"], 15.0)
        self.assertEqual(recorder.per_state[0][1]["state_id"], "B")


if __name__ == "__main__":
    unittest.main()
