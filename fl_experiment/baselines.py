"""Seasonal naive, SARIMA and rolling-origin local GRU on one saved fold file.

    python -m fl_experiment.baselines --order-grid small --epochs 20
    python -m fl_experiment.baselines --source panel --allow-assumed-vintages \
        --panel data/reports/lagged45-2012/panel.csv

Validation origins are scored. The test period is stored in the fold file and
not read by any model. SARIMA orders are chosen by AIC on data before the
first validation month, then frozen; each origin only refits those parameters.
A failed fit falls back to the seasonal naive and is counted.
"""

import argparse
import csv
import json
import math
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from statsmodels.tsa.statespace.sarimax import SARIMAX

from fl_experiment.folds import (
    DEFAULT_HOLDOUT,
    SEASONAL_PERIOD,
    build_folds,
    column_index,
    latest_before,
    origin_arrays,
    suffix_values,
    write_folds,
)
from fl_experiment.local_gru import (
    add_training_arguments,
    compute_info,
    train_arrays,
    training_settings,
)
from fl_experiment.metrics import mase, regression_metrics
from fl_experiment.server import MetricsRecorder
from fl_experiment.state_data import add_data_arguments, records_from_arguments, shift_month

RESULTS = Path(__file__).resolve().parent / "results"
NONSEASONAL = ((0, 1, 1), (1, 1, 0), (1, 1, 1), (0, 1, 0))
SEASONAL = ((0, 1, 0, SEASONAL_PERIOD), (0, 1, 1, SEASONAL_PERIOD), (1, 0, 0, SEASONAL_PERIOD))
GRIDS = {
    "full": [(order, seasonal) for order in NONSEASONAL for seasonal in SEASONAL],
    "small": [((1, 1, 0), (0, 1, 0, SEASONAL_PERIOD))],
}


def seasonal_naive(state, month):
    """Observed target one seasonal period before `month`."""
    lag = shift_month(month, -SEASONAL_PERIOD)
    values = dict(zip(state["months"], state["targets"]))
    if lag not in values:
        raise KeyError(lag)
    return values[lag]


def fit_sarima(values, order, seasonal_order, maxiter):
    """Fit one SARIMA and return the one-step-ahead forecast."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = SARIMAX(
            values,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=maxiter)
    forecast = float(np.asarray(result.forecast(1), dtype=float).reshape(-1)[0])
    return result, forecast


def select_state_order(state, val_start, grid, min_train, maxiter):
    """Pick the lowest-AIC order from months strictly before `val_start`."""
    end = latest_before(state, val_start)
    values = suffix_values(state, end) if end else []
    span = [end and shift_month(end, 1 - len(values)), end]
    best_score = math.inf
    best = {"order": None, "seasonal_order": None, "aic": None}
    if len(values) >= min_train:
        for order, seasonal in grid:
            try:
                result, _ = fit_sarima(values, order, seasonal, maxiter)
                score = float(result.aic)
            except Exception:  # pylint: disable=broad-exception-caught
                continue
            if math.isfinite(score) and score < best_score:
                best_score = score
                best = {"order": list(order), "seasonal_order": list(seasonal), "aic": score}
    best["selection_start"] = span[0]
    best["selection_end"] = span[1]
    best["selection_months"] = len(values)
    return best


def forecast_sarima(state, month, chosen, min_train, maxiter):
    """Refit the frozen order on months before `month`. Fall back to the seasonal naive."""
    naive = seasonal_naive(state, month)
    if not chosen["order"]:
        return naive, True
    values = suffix_values(state, shift_month(month, -1))
    if len(values) < min_train:
        return naive, True
    try:
        _, forecast = fit_sarima(values, chosen["order"], chosen["seasonal_order"], maxiter)
    except Exception:  # pylint: disable=broad-exception-caught
        return naive, True
    if not math.isfinite(forecast):
        return naive, True
    return forecast, False


def _prediction(model, state_id, origin, predicted, fallback=""):
    return {
        "model": model,
        "state_id": state_id,
        "target_month": origin["target_month"],
        "actual": origin["actual"],
        "predicted": predicted,
        "naive": origin["seasonal_naive"],
        "fallback": fallback,
    }


def naive_rows(folds):
    """One seasonal-naive prediction per scored origin."""
    rows = []
    for state_id, state in folds["states"].items():
        for origin in state["origins"]:
            if origin["scored"]:
                rows.append(_prediction(
                    "seasonal_naive", state_id, origin, origin["seasonal_naive"]
                ))
    return rows


def sarima_rows(folds, grid, maxiter):
    """Frozen-order SARIMA predictions, with naive fallbacks counted per state."""
    protocol = folds["protocol"]
    rows, orders = [], {}
    for state_id, state in folds["states"].items():
        chosen = select_state_order(
            state, protocol["val_start"], grid, protocol["min_train"], maxiter
        )
        fallbacks = 0
        for origin in state["origins"]:
            if not origin["scored"]:
                continue
            predicted, fell_back = forecast_sarima(
                state, origin["target_month"], chosen, protocol["min_train"], maxiter
            )
            fallbacks += int(fell_back)
            rows.append(_prediction(
                "sarima", state_id, origin, predicted, "seasonal_naive" if fell_back else ""
            ))
        chosen["fallbacks"] = fallbacks
        orders[state_id] = chosen
    return rows, orders


def _gru_forecast(state, origin, columns, protocol, settings):
    """Train on this origin's past and return the one-step prediction in INR crore."""
    arrays = origin_arrays(
        state, origin, protocol["window"], protocol["holdout_months"], columns
    )
    x_fit, y_fit = arrays["fit"]
    x_early, y_early = arrays["early_stop"]
    model, _ = train_arrays(
        torch.from_numpy(x_fit),
        torch.from_numpy(y_fit),
        torch.from_numpy(x_early),
        torch.from_numpy(y_early),
        len(columns),
        settings,
    )
    with torch.no_grad():
        x_origin, _ = arrays["origin"]
        normalized = model(torch.from_numpy(x_origin)).numpy().reshape(-1)[0]
    return float(np.expm1(normalized * arrays["y_std"] + arrays["y_mean"]))


def gru_rows(folds, feature_set, settings):
    """Retrain one GRU per scored origin. Early stopping uses the preceding holdout only."""
    protocol = folds["protocol"]
    columns = column_index(folds["feature_names"], feature_set)
    model_name = f"local_gru_{feature_set}"
    rows = []
    for index, (state_id, state) in enumerate(folds["states"].items()):
        seeded = dict(settings, seed=settings["seed"] + index)
        for origin in state["origins"]:
            if origin["scored"]:
                predicted = _gru_forecast(state, origin, columns, protocol, seeded)
                rows.append(_prediction(model_name, state_id, origin, predicted))
    return rows


def _state_summary(items):
    """Metrics for one state's scored origins."""
    actual = [item["actual"] for item in items]
    predicted = [item["predicted"] for item in items]
    naive = [item["naive"] for item in items]
    metrics = regression_metrics(actual, predicted, naive)
    metrics["origins"] = len(items)
    metrics["fallbacks"] = sum(bool(item["fallback"]) for item in items)
    return metrics, actual, predicted, naive


def _accumulate(grouped):
    """Collect per-state metrics and the pooled actual, predicted and naive values."""
    per_state, weighted = {}, []
    pooled = [[], [], [], []]
    for state_id, items in sorted(grouped.items()):
        metrics, actual, predicted, naive = _state_summary(items)
        per_state[state_id] = metrics
        weighted.append((len(items), metrics))
        pooled[0].extend(actual)
        pooled[1].extend(predicted)
        pooled[2].extend(naive)
        if math.isfinite(metrics["mase"]):
            pooled[3].append(metrics["mase"])
    return per_state, weighted, pooled


def summarize_model(rows):
    """Per-state and pooled validation metrics, including MASE against the seasonal naive."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["state_id"]].append(row)
    per_state, weighted, pooled = _accumulate(grouped)
    aggregate = MetricsRecorder()(weighted)
    aggregate["mase"] = mase(pooled[0], pooled[1], pooled[2])
    aggregate["mase_equal_state"] = float(np.mean(pooled[3])) if pooled[3] else float("nan")
    aggregate["origins"] = len(rows)
    aggregate["fallbacks"] = sum(item["fallbacks"] for item in per_state.values())
    return aggregate, per_state


def coverage(folds):
    """Count scored and skipped validation origins. Test origins are listed, not scored."""
    reasons = Counter()
    listed = scored = test = 0
    for state in folds["states"].values():
        for origin in state["origins"]:
            if origin["split"] == "test":
                test += 1
                continue
            listed += 1
            if origin["scored"]:
                scored += 1
            else:
                reasons[origin["skip_reason"]] += 1
    return {
        "validation_origins": listed,
        "scored_origins": scored,
        "skipped_validation_origins": dict(reasons),
        "stored_test_origins": test,
        "test_origins_scored": 0,
    }


def _same_origins(rows):
    """Every model must score the same state-months."""
    by_model = defaultdict(set)
    for row in rows:
        by_model[row["model"]].add((row["state_id"], row["target_month"]))
    origins = list(by_model.values())
    if origins and any(item != origins[0] for item in origins):
        raise RuntimeError("Models scored different origins")


def _write_predictions(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["model", "state_id", "target_month", "actual_crore", "predicted_crore", "fallback"]
        )
        for row in rows:
            writer.writerow([
                row["model"], row["state_id"], row["target_month"],
                f"{row['actual']:.4f}", f"{row['predicted']:.4f}", row["fallback"],
            ])


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--order-grid", choices=sorted(GRIDS), default="full")
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument("--gru-features", nargs="+", choices=["all", "gst_only"],
                        default=["all", "gst_only"])
    add_training_arguments(parser)
    parser.add_argument("--holdout", type=int, default=DEFAULT_HOLDOUT)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _model_rows(folds, args, settings):
    rows = naive_rows(folds)
    sarima, orders = sarima_rows(folds, GRIDS[args.order_grid], args.maxiter)
    rows.extend(sarima)
    for feature_set in args.gru_features:
        print(f"Training rolling-origin GRU ({feature_set})", flush=True)
        rows.extend(gru_rows(folds, feature_set, settings))
    return rows, orders


def _prepare(args):
    """Load records and build folds. The fold file keeps every input column."""
    records, features, val_start, test_start = records_from_arguments(args)
    folds = build_folds(
        records, features, val_start, test_start, args.min_train, args.window, args.holdout
    )
    return folds, test_start


def _model_summaries(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)
    summaries = {}
    for name, items in grouped.items():
        aggregate, per_state = summarize_model(items)
        summaries[name] = {"validation_aggregate": aggregate, "validation_by_state": per_state}
    return summaries


def main():
    """Build the folds, score validation origins, and write predictions and metrics."""
    args = parse_args()
    if args.features != "all":
        print("Folds keep every input; --features is ignored. Use --gru-features.", flush=True)
    args.features = "all"
    torch.set_num_threads(1)
    folds, test_start = _prepare(args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir or RESULTS / f"rolling-{args.source}-{stamp}"
    output.mkdir(parents=True, exist_ok=True)
    write_folds(folds, output / "folds.json")
    started = time.perf_counter()
    rows, orders = _model_rows(folds, args, training_settings(args))
    if any(row["target_month"] >= test_start for row in rows):
        raise RuntimeError("A test-period origin was scored")
    _same_origins(rows)
    _write_predictions(output / "predictions.csv", rows)
    record = {
        "created_utc": stamp,
        "protocol": folds["protocol"],
        "order_grid": args.order_grid,
        "gru_settings": training_settings(args),
        "seed_rule": "seed + state index, repeated at every origin",
        "source": args.source,
        "panel": args.panel,
        "synthetic": args.source == "synthetic",
        "assumed_lag": bool(args.allow_assumed_vintages),
        "excluded_states": folds["excluded_states"],
        "coverage": coverage(folds),
        "sarima_orders": orders,
        "models": _model_summaries(rows),
        "compute": compute_info(started),
        "limitation": (
            "Assumed-lag sensitivity on the validation origins only. "
            "Not a real-time backtest and not comparable to the fixed-split GRU run."
        ),
    }
    (output / "metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Scored origins: {record['coverage']['scored_origins']}", flush=True)
    for name, payload in record["models"].items():
        print(f"{name}: {json.dumps(payload['validation_aggregate'])}", flush=True)
    print(f"Outputs: {output}", flush=True)


if __name__ == "__main__":
    main()
