"""Rolling-origin folds, seasonal naive, SARIMA order selection and the local GRU."""
# pylint: disable=missing-function-docstring

import unittest

from fl_experiment.baselines import (
    GRIDS,
    forecast_sarima,
    gru_rows,
    naive_rows,
    select_state_order,
    summarize_model,
)
from fl_experiment.folds import build_folds, column_index, months_before, origin_arrays
from fl_experiment.metrics import mase
from fl_experiment.tests.test_state_clients import monthly_records

FEATURES = ["gst_previous_month", "gst_same_month_last_year"]
GRU = {"epochs": 3, "patience": 2, "lr": 0.05, "weight_decay": 0.0, "hidden": 8,
       "dropout": 0.0, "seed": 0}


def sample_folds(records=None, val_start="2021-01", test_start="2022-01"):
    """One state of complete months, with the fixed validation and test cutoffs."""
    records = records if records is not None else monthly_records("A", "2018-01", 60)
    return build_folds(records, FEATURES, val_start, test_start)


class FoldTests(unittest.TestCase):
    """Shared expanding-window origins."""

    def test_validation_is_scored_and_test_is_only_stored(self):
        folds = sample_folds()
        origins = folds["states"]["A"]["origins"]
        scored = [origin for origin in origins if origin["scored"]]
        self.assertEqual([origin["target_month"] for origin in scored],
                         [f"2021-{month:02d}" for month in range(1, 13)])
        test = [origin for origin in origins if origin["split"] == "test"]
        self.assertTrue(test)
        self.assertTrue(all(not origin["scored"] for origin in test))
        for origin in origins:
            self.assertLess(origin["train_last"], origin["target_month"])
            self.assertNotIn(origin["target_month"], months_before(folds["states"]["A"],
                                                                    origin["target_month"]))

    def test_windows_used_for_an_origin_stop_before_it(self):
        folds = sample_folds()
        state = folds["states"]["A"]
        origin = next(item for item in state["origins"] if item["target_month"] == "2021-06")
        arrays = origin_arrays(state, origin, 6, 6, column_index(FEATURES, "all"))
        self.assertLess(max(arrays["fit_months"]), min(arrays["early_months"]))
        self.assertLess(max(arrays["early_months"]), origin["target_month"])
        self.assertEqual(len(arrays["early_months"]), 6)

    def test_missing_seasonal_lag_is_skipped_not_imputed(self):
        records = [row for row in monthly_records("A", "2018-01", 60) if row[1] != "2020-01"]
        folds = sample_folds(records)
        origin = next(item for item in folds["states"]["A"]["origins"]
                      if item["target_month"] == "2021-01")
        self.assertFalse(origin["scored"])
        self.assertIn("seasonal lag", origin["skip_reason"])
        scored = {(row["state_id"], row["target_month"]) for row in naive_rows(folds)}
        self.assertNotIn(("A", "2021-01"), scored)

    def test_gst_only_keeps_the_gst_columns_and_month_encoding(self):
        names = ["feature_gst_excluding_cess_crore", "feature_ewb_intra_bills"]
        self.assertEqual(column_index(names, "gst_only"), [0, 2, 3])
        self.assertEqual(column_index(names, "all"), [0, 1, 2, 3])


class BaselineTests(unittest.TestCase):
    """Forecasts may use only information from before the origin."""

    def test_seasonal_naive_is_last_year_and_has_mase_one(self):
        folds = sample_folds()
        rows = naive_rows(folds)
        values = dict(zip(folds["states"]["A"]["months"], folds["states"]["A"]["targets"]))
        for row in rows:
            year, month = row["target_month"].split("-")
            self.assertEqual(row["predicted"], values[f"{int(year) - 1:04d}-{month}"])
        aggregate, _ = summarize_model(rows)
        self.assertAlmostEqual(aggregate["mase"], 1.0)
        self.assertAlmostEqual(mase([1.0, 2.0], [2.0, 3.0], [0.0, 1.0]), 1.0)

    def test_sarima_ignores_the_origin_value(self):
        folds = sample_folds()
        state = folds["states"]["A"]
        chosen = select_state_order(state, "2021-01", GRIDS["small"], 24, 50)
        self.assertIsNotNone(chosen["order"])
        forecast, fell_back = forecast_sarima(state, "2021-06", chosen, 24, 50)
        self.assertFalse(fell_back)
        index = state["months"].index("2021-06")
        state["targets"][index] = 99999.0
        again = select_state_order(state, "2021-01", GRIDS["small"], 24, 50)
        revised, _ = forecast_sarima(state, "2021-06", chosen, 24, 50)
        self.assertEqual(chosen["order"], again["order"])
        self.assertEqual(chosen["aic"], again["aic"])
        self.assertEqual(forecast, revised)

    def test_rolling_gru_scores_validation_origins_only(self):
        folds = sample_folds()
        rows = gru_rows(folds, "all", GRU)
        scored = [origin["target_month"] for origin in folds["states"]["A"]["origins"]
                  if origin["scored"]]
        self.assertEqual([row["target_month"] for row in rows], scored)
        self.assertTrue(all(row["target_month"] < "2022-01" for row in rows))
        self.assertTrue(all(not row["fallback"] for row in rows))


if __name__ == "__main__":
    unittest.main()
