"""Client ownership, chronology, preprocessing and Flower client behaviour."""
# pylint: disable=missing-function-docstring

import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

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
    shift_month,
    write_cache,
)


def monthly_records(state, start, count, base=100.0):
    """Synthetic-shaped records with one feature pair per target month."""
    records = []
    for offset in range(count):
        month = shift_month(start, offset)
        records.append((state, month, [base + offset, base + offset - 12], base + offset + 1))
    return records


class StateClientTests(unittest.TestCase):
    """Behaviour required by issue #18."""

    def test_synthetic_clients_are_disjoint_states_with_ordered_splits(self):
        records, _ = load_synthetic_records()
        clients, excluded = build_state_clients(records, "2022-01", "2023-01")
        self.assertEqual(len(clients), 10)
        self.assertEqual(excluded, {})
        self.assertEqual(len({c.state_id for c in clients}), len(clients))
        for client in clients:
            self.assertLess(max(client.months["train"]), min(client.months["val"]))
            self.assertLess(max(client.months["val"]), min(client.months["test"]))

    def test_every_record_is_owned_by_exactly_one_client(self):
        records, _ = load_synthetic_records()
        clients, _ = build_state_clients(records, "2022-01", "2023-01")
        owned = [
            (client.state_id, month)
            for client in clients
            for months in client.months.values()
            for month in months
        ]
        self.assertEqual(sorted(owned), sorted((r[0], r[1]) for r in records))

    def test_synthetic_inputs_exclude_target_derived_columns(self):
        _, features = load_synthetic_records()
        self.assertNotIn("sgst_share", features)
        self.assertNotIn("cgst_share", features)

    def test_scaling_uses_training_rows_only(self):
        records = monthly_records("A", "2019-01", 48)
        changed = [
            (s, m, [f * 1000 for f in x], y * 1000) if m >= "2021-01" else (s, m, x, y)
            for s, m, x, y in records
        ]
        first, _ = build_state_clients(records, "2021-01", "2022-01")
        second, _ = build_state_clients(changed, "2021-01", "2022-01")
        np.testing.assert_allclose(first[0].x_mean, second[0].x_mean)
        self.assertAlmostEqual(first[0].y_mean, second[0].y_mean)
        np.testing.assert_allclose(first[0].x["train"].mean(axis=0)[:2], 0.0, atol=1e-5)

    def test_short_or_unevaluable_states_are_excluded_with_reason(self):
        records = monthly_records("A", "2019-01", 48) + monthly_records("B", "2020-06", 10)
        records += monthly_records("C", "2018-01", 30)
        clients, excluded = build_state_clients(records, "2021-01", "2022-01")
        self.assertEqual([c.state_id for c in clients], ["A"])
        self.assertIn("training rows", excluded["B"])
        self.assertEqual(excluded["C"], "no validation or test rows")

    def test_duplicate_state_month_is_rejected(self):
        records = monthly_records("A", "2019-01", 48)
        with self.assertRaises(ValueError):
            build_state_clients(records + records[:1], "2021-01", "2022-01")

    def test_check_disjoint_rejects_repeated_state(self):
        clients, _ = build_state_clients(monthly_records("A", "2019-01", 48), "2021-01", "2022-01")
        with self.assertRaises(ValueError):
            check_disjoint(clients + clients)

    def test_decode_inverts_target_transform(self):
        clients, _ = build_state_clients(monthly_records("A", "2019-01", 48), "2021-01", "2022-01")
        client = clients[0]
        expected = [r[3] for r in monthly_records("A", "2019-01", 48) if r[1] >= "2022-01"]
        np.testing.assert_allclose(client.decode(client.y["test"]), expected, rtol=1e-4)

    def test_cache_round_trip(self):
        clients, _ = build_state_clients(monthly_records("A", "2019-01", 48), "2021-01", "2022-01")
        with tempfile.TemporaryDirectory() as folder:
            write_cache(clients, folder, {"source": "test"})
            loaded = load_cached_client(folder, 0)
        self.assertEqual(loaded.state_id, "A")
        self.assertEqual(loaded.months, clients[0].months)
        np.testing.assert_array_equal(loaded.x["val"], clients[0].x["val"])

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

    def test_client_fit_and_evaluate(self):
        clients, _ = build_state_clients(monthly_records("A", "2019-01", 48), "2021-01", "2022-01")
        client = StateClient(clients[0], local_epochs=20, lr=0.05)
        initial = client.get_parameters({})
        _, _, before = client.evaluate(initial, {})
        updated, count, fit_metrics = client.fit(initial, {})
        self.assertEqual(count, 24)
        self.assertEqual([p.shape for p in updated], [p.shape for p in initial])
        self.assertTrue(math.isfinite(fit_metrics["train_loss"]))
        loss, count, metrics = client.evaluate(updated, {})
        self.assertEqual(count, 12)
        self.assertEqual(metrics["state_id"], "A")
        self.assertTrue(math.isfinite(loss))
        self.assertEqual(set(before) - {"state_id"}, {"rmse", "mae", "mape"})

    def test_recorder_reports_weighted_and_equal_state_means(self):
        recorder = MetricsRecorder()
        result = recorder([(1, {"rmse": 10.0, "mae": 1.0, "mape": 1.0, "state_id": "A"}),
                           (3, {"rmse": 20.0, "mae": 1.0, "mape": 1.0, "state_id": "B"})])
        self.assertAlmostEqual(result["rmse"], 17.5)
        self.assertAlmostEqual(result["rmse_equal_state"], 15.0)
        self.assertEqual(recorder.per_state[0][1]["state_id"], "B")


if __name__ == "__main__":
    unittest.main()
