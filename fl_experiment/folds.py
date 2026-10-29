"""Rolling-origin folds shared by the seasonal naive, SARIMA and local GRU.

Horizon is one month. For target month m, every model may use only months
before m. Origins in [val_start, test_start) are scored. Later origins are
stored and never scored. An origin is scored only when the seasonal lag
exists, history is long enough, and the GRU can form its training windows,
early-stopping holdout and origin window without crossing a gap.
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from fl_experiment.state_data import (
    _raw_inputs,
    _stack,
    _training_scales,
    shift_month,
    window_targets,
)

HORIZON = 1
SEASONAL_PERIOD = 12
DEFAULT_HOLDOUT = 6


def column_index(feature_names, feature_set):
    """Input columns for a GRU feature set. The last two columns are the month encoding."""
    width = len(feature_names) + 2
    if feature_set == "all":
        return list(range(width))
    if feature_set != "gst_only":
        raise ValueError("feature_set must be 'all' or 'gst_only'")
    gst = [index for index, name in enumerate(feature_names) if "gst" in name]
    if not gst:
        raise ValueError("gst_only requires a GST input")
    return gst + [width - 2, width - 1]


def _holdout_months(month, holdout):
    return [shift_month(month, -offset) for offset in range(holdout, 0, -1)]


def _skip_reason(month, months, usable, min_train, holdout):
    """Return why this origin cannot be scored, or an empty string."""
    index = months.index(month)
    fit_end = shift_month(month, -holdout)
    fit_windows = sum(target < fit_end for target in usable)
    holdout_missing = any(target not in usable for target in _holdout_months(month, holdout))
    checks = (
        (index < min_train, f"{index} training months < {min_train}"),
        (shift_month(month, -SEASONAL_PERIOD) not in months, "seasonal lag missing"),
        (fit_windows < min_train, f"{fit_windows} fitting windows < {min_train}"),
        (holdout_missing, "early-stopping holdout incomplete"),
        (month not in usable, "origin window incomplete"),
        (shift_month(month, -1) not in months, "month before origin missing"),
    )
    return next((reason for failed, reason in checks if failed), "")


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def _origin(month, months, targets, usable, test_start, min_train, holdout):
    reason = _skip_reason(month, months, usable, min_train, holdout)
    split = "val" if month < test_start else "test"
    index = months.index(month)
    fit_end = shift_month(month, -holdout)
    lag = shift_month(month, -SEASONAL_PERIOD)
    return {
        "target_month": month,
        "split": split,
        "actual": targets[month],
        "train_last": months[index - 1] if index else None,
        "n_history": index,
        "n_fit_windows": sum(target < fit_end for target in usable),
        "seasonal_naive": targets.get(lag),
        "scored": split == "val" and reason == "",
        "skip_reason": reason,
    }


def _state_entry(rows, val_start, test_start, min_train, window, holdout):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    rows = sorted(rows, key=lambda record: record[1])
    if len({row[1] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate target month for state {rows[0][0]}")
    months = [row[1] for row in rows]
    targets = {row[1]: row[3] for row in rows}
    inputs = {row[1]: _raw_inputs(row[2], row[1]) for row in rows}
    usable = window_targets(months, window)
    origins = [
        _origin(month, months, targets, usable, test_start, min_train, holdout)
        for month in months
        if month >= val_start
    ]
    return {
        "months": months,
        "targets": [targets[month] for month in months],
        "inputs": [inputs[month] for month in months],
        "origins": origins,
    }


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def build_folds(records, feature_names, val_start, test_start, min_train=24, window=6,
                holdout=DEFAULT_HOLDOUT):
    """Return the fold document. Test origins are present and have scored=False."""
    if not val_start < test_start:
        raise ValueError("Validation start must precede test start")
    if window < 1 or holdout < 1:
        raise ValueError("window and holdout must be at least 1")
    by_state = defaultdict(list)
    for record in records:
        by_state[record[0]].append(record)
    states, excluded = {}, {}
    for state in sorted(by_state, key=lambda name: (len(name), name)):
        entry = _state_entry(
            by_state[state], val_start, test_start, min_train, window, holdout
        )
        if not entry["origins"]:
            continue
        states[state] = entry
        scored = [origin for origin in entry["origins"] if origin["scored"]]
        if not scored:
            skipped = [origin for origin in entry["origins"] if origin["split"] == "val"]
            excluded[state] = skipped[0]["skip_reason"] if skipped else "no validation origin"
    return {
        "protocol": {
            "horizon_months": HORIZON,
            "seasonal_period": SEASONAL_PERIOD,
            "val_start": val_start,
            "test_start": test_start,
            "min_train": min_train,
            "window": window,
            "holdout_months": holdout,
            "expanding_window": True,
            "test_scored": False,
        },
        "feature_names": list(feature_names),
        "states": states,
        "excluded_states": excluded,
    }


def write_folds(folds, path):
    """Write the fold document as JSON."""
    Path(path).write_text(json.dumps(folds), encoding="utf-8")
    return Path(path)


def months_before(state, month):
    """Months of this state strictly before `month`."""
    return [item for item in state["months"] if item < month]


def suffix_values(state, end_month):
    """Contiguous raw target values ending at `end_month`, oldest first.

    A gap stops the walk. The returned series never contains `end_month`'s
    successor, so a one-step forecast is the month after `end_month`.
    """
    values = dict(zip(state["months"], state["targets"]))
    if end_month not in values:
        return []
    month, collected = end_month, []
    while month in values:
        collected.append(values[month])
        month = shift_month(month, -1)
    collected.reverse()
    return collected


def latest_before(state, cutoff):
    """Latest stored month strictly before `cutoff`."""
    earlier = months_before(state, cutoff)
    return earlier[-1] if earlier else None


def _sliced_inputs(state, columns):
    return {
        month: [row[index] for index in columns]
        for month, row in zip(state["months"], state["inputs"])
    }


def _window_block(inputs, log_targets, target_months, window, scales):
    """Normalize one block of windows with scales fitted elsewhere."""
    x_mean, x_std, y_mean, y_std = scales
    raw = _stack(inputs, target_months, window, len(x_mean))
    logs = [log_targets[target] for target in target_months]
    values = np.asarray(logs, dtype=float).reshape(-1, 1)
    return (
        ((raw - x_mean) / x_std).astype(np.float32),
        ((values - y_mean) / y_std).astype(np.float32),
    )


def pooled_scales(states, fit_end, columns):
    """Mean and standard deviation of every state's rows strictly before `fit_end`."""
    rows, logs = [], []
    for state in states:
        inputs = _sliced_inputs(state, columns)
        log_targets = {
            month: float(np.log1p(value))
            for month, value in zip(state["months"], state["targets"])
        }
        for month, row in inputs.items():
            if month < fit_end:
                rows.append(row)
                logs.append(log_targets[month])
    values = np.asarray(rows, dtype=float)
    target = np.asarray(logs, dtype=float).reshape(-1, 1)
    spread = values.std(axis=0)
    target_spread = target.std(axis=0)
    return (
        values.mean(axis=0),
        np.where(spread < 1e-8, 1.0, spread),
        float(target.mean()),
        float(np.where(target_spread < 1e-8, 1.0, target_spread)[0]),
    )


def protocol_hash(folds):
    """Fingerprint of the folds. The test command refuses a different fingerprint."""
    origins = []
    for state_id, state in sorted(folds["states"].items()):
        for origin in state["origins"]:
            origins.append({
                "state_id": state_id,
                "target_month": origin["target_month"],
                "split": origin["split"],
                "actual": round(float(origin["actual"]), 6),
                "scored": bool(origin["scored"]),
                "skip_reason": origin["skip_reason"],
            })
    payload = {
        "protocol": folds["protocol"],
        "feature_names": folds["feature_names"],
        "origins": origins,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def origin_arrays(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        state, origin, window, holdout, columns, scales=None):
    """Scaled fit, early-stopping and origin arrays. Nothing uses the origin month."""
    month = origin["target_month"]
    inputs = _sliced_inputs(state, columns)
    log_targets = {
        item: float(np.log1p(value))
        for item, value in zip(state["months"], state["targets"])
    }
    fit_end = shift_month(month, -holdout)
    usable = window_targets(inputs, window)
    fit_months = [target for target in usable if target < fit_end]
    early_months = [target for target in usable if fit_end <= target < month]
    if month in fit_months or month in early_months:
        raise ValueError("Origin month leaked into training or early stopping")
    if scales is None:
        scales = _training_scales(inputs, log_targets, fit_end)
    return {
        "fit": _window_block(inputs, log_targets, fit_months, window, scales),
        "early_stop": _window_block(inputs, log_targets, early_months, window, scales),
        "origin": _window_block(inputs, log_targets, [month], window, scales),
        "y_mean": scales[2],
        "y_std": scales[3],
        "fit_months": fit_months,
        "early_months": early_months,
    }
