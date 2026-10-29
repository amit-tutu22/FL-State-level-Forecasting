"""Score the test origins once, using a frozen selection.json.

    python -m fl_experiment.evaluate_test --confirm-frozen \
        --selection docs/experiments/2026-10-29-selection.json \
        --source panel --allow-assumed-vintages \
        --panel data/reports/lagged45-2012/panel.csv

The command refuses to start when the selection file is missing, not frozen,
or hashed against different folds. It does not search feature sets, mu, or
the APFL step again. Training for a test month may use earlier months,
including the validation year.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from fl_experiment.baselines import GRIDS, select_state_order
from fl_experiment.folds import build_folds, protocol_hash
from fl_experiment.local_gru import compute_info
from fl_experiment.metrics import prediction_row
from fl_experiment.reporting import collate, ranked_states, summarize_model, write_json, write_rows
from fl_experiment.rolling import federated_rows, local_rows, pooled_rows
from fl_experiment.state_data import add_data_arguments, records_from_arguments

RESULTS = Path(__file__).resolve().parent / "results"


def require_selection(selection, folds):
    """Stop unless this file is the frozen match for these folds."""
    if not selection.get("frozen"):
        raise SystemExit("selection.json is not frozen")
    current = protocol_hash(folds)
    recorded = selection.get("protocol_hash")
    if recorded != current:
        raise SystemExit(
            "Fold hash does not match selection.json "
            f"(file {recorded}, current {current})"
        )


def add_second_look_arguments(parser):
    """--selection, --split and --second-look, for runners that reuse frozen settings."""
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--second-look", action="store_true",
                        help="required for --split test: the test year was already scored")


def frozen_folds(args):
    """Apply the test-year gate, then return the frozen selection and its matching folds."""
    if args.split == "test" and not args.second_look:
        raise SystemExit("The test year was already scored. Pass --second-look to re-score it.")
    if not args.selection.is_file():
        raise SystemExit(f"selection file not found: {args.selection}")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    records, feature_names, val_start, test_start = records_from_arguments(args)
    folds = build_folds(records, feature_names, val_start, test_start, args.min_train, args.window)
    require_selection(selection, folds)
    return selection, folds


def settings_from(selection, seed):
    """Training settings stored at the freeze. The seed is the only change."""
    return {
        **selection["architecture"],
        "seed": seed,
        "local_epochs": selection["local_epochs"],
        "max_rounds": selection["max_rounds"],
        "round_patience": selection["round_patience"],
    }


def _reference_rows(folds):
    """Seasonal naive and the SARIMA orders frozen before the validation year."""
    rows = []
    for state_id, state in folds["states"].items():
        for origin in state["origins"]:
            if origin["split"] != "test" or origin["skip_reason"]:
                continue
            row = prediction_row(
                "seasonal_naive", state_id, origin, origin["seasonal_naive"]
            )
            row["seed"] = 0
            rows.append(row)
    return rows


def _sarima_test_rows(folds, maxiter):
    """Refit frozen orders on each test origin. Orders are not searched again."""
    from fl_experiment.baselines import forecast_sarima  # pylint: disable=import-outside-toplevel
    protocol = folds["protocol"]
    rows = []
    for state_id, state in folds["states"].items():
        chosen = select_state_order(
            state, protocol["val_start"], GRIDS["full"], protocol["min_train"], maxiter
        )
        for origin in state["origins"]:
            if origin["split"] != "test" or origin["skip_reason"]:
                continue
            predicted, fell_back = forecast_sarima(
                state, origin["target_month"], chosen, protocol["min_train"], maxiter
            )
            row = prediction_row(
                "sarima", state_id, origin, predicted,
                "seasonal_naive" if fell_back else "",
            )
            row["seed"] = 0
            rows.append(row)
    return rows


def _neural(folds, selection):
    """Every neural method, with the settings chosen on validation."""
    feature_set = selection["feature_set"]
    rows = []
    jobs = [
        ("local", lambda settings: local_rows(folds, feature_set, settings, "test")),
        ("pooled", lambda settings: pooled_rows(folds, feature_set, settings, "test")),
        ("fedavg", lambda settings: federated_rows(folds, feature_set, settings, "test", "fedavg")),
        ("fedprox", lambda settings: federated_rows(
            folds, feature_set, settings, "test", "fedprox", selection["fedprox_mu"]
        )),
        ("apfl", lambda settings: federated_rows(
            folds, feature_set, settings, "test", "apfl", selection["apfl_alpha_step"]
        )),
    ]
    for label, runner in jobs:
        print(label, flush=True)
        for seed in selection["seeds"]:
            print(f"  seed {seed}", flush=True)
            rows.extend(runner(settings_from(selection, seed)))
    return rows


def _summarize(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["model"], row["seed"]), []).append(row)
    summaries, by_state = {}, {}
    for (model, _), items in sorted(grouped.items()):
        aggregate, per_state = summarize_model(items)
        summaries.setdefault(model, []).append(aggregate)
        by_state.setdefault(model, []).append(per_state)
    published, states = {}, {}
    for model, aggregates in summaries.items():
        published[model], states[model] = collate(list(zip(aggregates, by_state[model])))
    return published, states


def parse_args():
    """Parse the gated test options. There is no hyperparameter grid."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--confirm-frozen", action="store_true")
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main():
    """Score test origins and write the test table inputs."""
    args = parse_args()
    if not args.confirm_frozen:
        raise SystemExit("Refusing to score the test period without --confirm-frozen")
    if not args.selection.is_file():
        raise SystemExit(f"selection file not found: {args.selection}")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    started = time.perf_counter()
    records, feature_names, val_start, test_start = records_from_arguments(args)
    folds = build_folds(records, feature_names, val_start, test_start, args.min_train, args.window)
    require_selection(selection, folds)
    rows = _reference_rows(folds)
    rows.extend(_sarima_test_rows(folds, args.maxiter))
    rows.extend(_neural(folds, selection))
    if any(row["target_month"] < folds["protocol"]["test_start"] for row in rows):
        raise SystemExit("A test run produced a row before the test period")
    summaries, by_state = _summarize(rows)
    output = args.output_dir or RESULTS / "protocol-test"
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "predictions.csv", rows)
    write_json(output / "metrics.json", {
        "protocol_hash": protocol_hash(folds),
        "selection": selection,
        "summaries": summaries,
        "largest_rmse": {name: ranked_states(states, "rmse") for name, states in by_state.items()},
        "largest_mape": {name: ranked_states(states, "mape") for name, states in by_state.items()},
        "per_state": by_state,
        "compute": compute_info(started),
    })
    print(f"Wrote {output / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
