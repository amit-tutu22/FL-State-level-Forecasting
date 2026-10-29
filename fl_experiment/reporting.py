"""Shared summaries for the validation freeze and the gated test run."""

import csv
import json
from collections import defaultdict

import numpy as np

from fl_experiment.baselines import summarize_model

METRIC_KEYS = (
    "rmse_equal_state", "rmse", "mae_equal_state", "mae",
    "mape_equal_state", "mape", "mase_equal_state", "mase",
)


def write_json(path, payload):
    """Write one JSON document."""
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_rows(path, rows):
    """Write prediction rows as CSV."""
    fields = [
        "model", "seed", "state_id", "target_month",
        "actual", "predicted", "naive", "fallback",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean_std(values):
    array = np.asarray(values, dtype=float)
    spread = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    return {"mean": float(array.mean()), "std": spread}


def across_seeds(aggregates):
    """Mean and standard deviation of each summary metric across seeds."""
    return {key: _mean_std([item[key] for item in aggregates]) for key in METRIC_KEYS}


def per_state_across_seeds(per_states):
    """Per-state RMSE and MAPE across seeds."""
    report = {}
    for state_id in per_states[0]:
        report[state_id] = {
            "rmse": _mean_std([item[state_id]["rmse"] for item in per_states]),
            "mape": _mean_std([item[state_id]["mape"] for item in per_states]),
            "mae": _mean_std([item[state_id]["mae"] for item in per_states]),
            "mase": _mean_std([item[state_id]["mase"] for item in per_states]),
        }
    return report


def collate(seed_runs):
    """Turn one (aggregate, per_state) pair per seed into the published summary."""
    summary = across_seeds([item[0] for item in seed_runs])
    by_state = per_state_across_seeds([item[1] for item in seed_runs])
    return summary, by_state


def ranked_states(by_state, key, limit=5):
    """States with the largest mean of `key` ('rmse' or 'mape')."""
    ordered = sorted(by_state, key=lambda state_id: by_state[state_id][key]["mean"], reverse=True)
    return [
        {"state_id": state_id, **by_state[state_id][key]}
        for state_id in ordered[:limit]
    ]


def model_summaries(rows):
    """Seed-collated summaries, largest-RMSE states and per-state tables, by model."""
    summaries, by_state = {}, {}
    for model, runs in summarize_rows(rows).items():
        summaries[model], by_state[model] = collate([(item[1], item[2]) for item in runs])
    return {
        "summaries": summaries,
        "largest_rmse": {name: ranked_states(states, "rmse") for name, states in by_state.items()},
        "per_state": by_state,
    }


def summarize_rows(rows):
    """summarize_model for rows that may carry several models and seeds."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row.get("seed", 0))].append(row)
    report = {}
    for (model, seed), items in sorted(grouped.items()):
        aggregate, per_state = summarize_model(items)
        report.setdefault(model, []).append((seed, aggregate, per_state))
    return report
