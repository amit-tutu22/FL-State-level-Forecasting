"""Choose the shared feature set, FedProx mu and APFL step on validation only.

    python -m fl_experiment.compare --source panel --allow-assumed-vintages \
        --panel data/reports/lagged45-2012/panel.csv \
        --output-dir fl_experiment/results/real-2026-10-29/protocol

The score is the mean, across seeds, of equal-state RMSE in INR crore.
The test period is not read. See docs/PROTOCOL.md.
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from fl_experiment.baselines import GRIDS, naive_rows, sarima_rows, summarize_model
from fl_experiment.federated import (
    ALPHA_INIT, ALPHA_STEPS, LOCAL_EPOCHS, MAX_ROUNDS, MU_GRID, ROUND_PATIENCE,
)
from fl_experiment.folds import build_folds, protocol_hash, write_folds
from fl_experiment.local_gru import add_training_arguments, compute_info, training_settings
from fl_experiment.reporting import collate, ranked_states, summarize_rows, write_json, write_rows
from fl_experiment.rolling import federated_rows, local_rows, pooled_rows
from fl_experiment.state_data import add_data_arguments, records_from_arguments

RESULTS = Path(__file__).resolve().parent / "results"


def parse_args():
    """Parse the validation-comparison options."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    add_training_arguments(parser)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--feature-sets", nargs="+", default=["gst_only", "all"])
    parser.add_argument("--mu-grid", type=float, nargs="+", default=list(MU_GRID))
    parser.add_argument("--alpha-steps", type=float, nargs="+", default=list(ALPHA_STEPS))
    parser.add_argument("--local-epochs", type=int, default=LOCAL_EPOCHS)
    parser.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    parser.add_argument("--round-patience", type=int, default=ROUND_PATIENCE)
    parser.add_argument("--order-grid", choices=("full", "small"), default="full")
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument("--skip-references", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _settings(args, seed):
    settings = training_settings(args)
    settings.update({
        "seed": seed,
        "local_epochs": args.local_epochs,
        "max_rounds": args.max_rounds,
        "round_patience": args.round_patience,
    })
    return settings


def _run_seeds(args, runner):
    """Fit `runner(settings)` once per seed and collate the summaries."""
    runs, rows = [], []
    for seed in args.seeds:
        print(f"  seed {seed}", flush=True)
        batch = runner(_settings(args, seed))
        rows.extend(batch)
        runs.append(summarize_model(batch))
    summary, by_state = collate(runs)
    return rows, summary, by_state


def _feature_choice(args, folds):
    """Pick the feature set from the local GRU, then keep its predictions."""
    candidates, chosen_rows, chosen_states = {}, [], {}
    for feature_set in args.feature_sets:
        print(f"local GRU {feature_set}", flush=True)
        rows, summary, by_state = _run_seeds(
            args, lambda settings, feature_set=feature_set: local_rows(
                folds, feature_set, settings, "val"
            ),
        )
        candidates[feature_set] = summary
        chosen_rows.append(rows)
        chosen_states[feature_set] = by_state
        print(f"  equal-state RMSE {summary['rmse_equal_state']['mean']:.2f}", flush=True)
    selected = min(candidates, key=lambda name: candidates[name]["rmse_equal_state"]["mean"])
    print(f"selected feature set: {selected}", flush=True)
    return selected, candidates, chosen_rows, chosen_states


def _method_choice(args, folds, feature_set):
    """Fit pooled GRU, FedAvg, the mu grid and the APFL step grid."""
    rows, summaries, states = [], {}, {}

    def remember(name, batch, summary, by_state):
        rows.extend(batch)
        summaries[name] = summary
        states[name] = by_state
        print(f"  {name} equal-state RMSE {summary['rmse_equal_state']['mean']:.2f}", flush=True)

    print("pooled GRU", flush=True)
    remember("pooled_gru", *_run_seeds(
        args, lambda settings: pooled_rows(folds, feature_set, settings, "val")
    ))
    print("FedAvg", flush=True)
    remember("fedavg", *_run_seeds(
        args, lambda settings: federated_rows(folds, feature_set, settings, "val", "fedavg")
    ))
    for mu in args.mu_grid:
        print(f"FedProx mu={mu:g}", flush=True)
        name = f"fedprox_mu_{mu:g}"
        remember(name, *_run_seeds(
            args,
            lambda settings, mu=mu: federated_rows(
                folds, feature_set, settings, "val", "fedprox", mu
            ),
        ))
    for step in args.alpha_steps:
        print(f"APFL step={step:g}", flush=True)
        name = f"apfl_step_{step:g}"
        remember(name, *_run_seeds(
            args,
            lambda settings, step=step: federated_rows(
                folds, feature_set, settings, "val", "apfl", step
            ),
        ))
    chosen_mu = min(
        args.mu_grid,
        key=lambda mu: summaries[f"fedprox_mu_{mu:g}"]["rmse_equal_state"]["mean"],
    )
    chosen_step = min(
        args.alpha_steps,
        key=lambda step: summaries[f"apfl_step_{step:g}"]["rmse_equal_state"]["mean"],
    )
    return rows, summaries, states, chosen_mu, chosen_step


def _references(args, folds):
    """Seasonal naive and frozen-order SARIMA. They are not retuned."""
    if args.skip_references:
        return [], {}
    rows = []
    for row in naive_rows(folds):
        row["seed"] = 0
        rows.append(row)
    sarima, orders = sarima_rows(folds, GRIDS[args.order_grid], args.maxiter)
    for row in sarima:
        row["seed"] = 0
        rows.append(row)
    return rows, orders


def _selection(args, folds, choice, summaries, started):
    settings = _settings(args, args.seeds[0])
    return {
        "frozen": True,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol_hash": protocol_hash(folds),
        "protocol": folds["protocol"],
        "selection_metric": "rmse_equal_state",
        "feature_set": choice["feature_set"],
        "fedprox_mu": choice["mu"],
        "apfl_alpha_step": choice["step"],
        "apfl_alpha_init": ALPHA_INIT,
        "seeds": list(args.seeds),
        "local_epochs": args.local_epochs,
        "max_rounds": args.max_rounds,
        "round_patience": args.round_patience,
        "architecture": {
            "epochs": settings["epochs"],
            "patience": settings["patience"],
            "lr": settings["lr"],
            "weight_decay": settings["weight_decay"],
            "hidden": settings["hidden"],
            "dropout": settings["dropout"],
        },
        "validation_rmse_equal_state": {
            name: summaries[name]["rmse_equal_state"] for name in summaries
        },
        "compute": compute_info(started),
    }


def _merge(feature_summaries, feature_states, method_summaries, method_states):
    """Prefix the local-GRU names and add the other methods."""
    summaries = {
        f"local_gru_{name}": value for name, value in feature_summaries.items()
    }
    summaries.update(method_summaries)
    by_state = {f"local_gru_{name}": value for name, value in feature_states.items()}
    by_state.update(method_states)
    return summaries, by_state


def _collect(args, folds):
    """Run every validation candidate and return rows plus the chosen settings."""
    reference_rows, orders = _references(args, folds)
    feature = _feature_choice(args, folds)
    method = _method_choice(args, folds, feature[0])
    summaries, by_state = _merge(feature[1], feature[3], method[1], method[2])
    if reference_rows:
        for model, runs in summarize_rows(reference_rows).items():
            pair = [(item[1], item[2]) for item in runs]
            summaries[model], by_state[model] = collate(pair)
    rows = [row for batch in feature[2] for row in batch]
    rows.extend(method[0])
    rows.extend(reference_rows)
    return {
        "rows": rows,
        "summaries": summaries,
        "by_state": by_state,
        "orders": orders,
        "choice": {"feature_set": feature[0], "mu": method[3], "step": method[4]},
    }


def main():
    """Run the validation comparison and write selection.json."""
    args = parse_args()
    if args.features != "all":
        raise SystemExit("Folds keep every input. Choose GRU features with --feature-sets.")
    torch.set_num_threads(1)
    started = time.perf_counter()
    records, feature_names, val_start, test_start = records_from_arguments(args)
    folds = build_folds(records, feature_names, val_start, test_start, args.min_train, args.window)
    output = args.output_dir or RESULTS / "protocol-validation"
    output.mkdir(parents=True, exist_ok=True)
    write_folds(folds, output / "folds.json")
    collected = _collect(args, folds)
    selection = _selection(args, folds, collected["choice"], collected["summaries"], started)
    by_state = collected["by_state"]
    metrics = {
        "selection": selection,
        "summaries": collected["summaries"],
        "largest_rmse": {
            name: ranked_states(states, "rmse") for name, states in by_state.items()
        },
        "largest_mape": {
            name: ranked_states(states, "mape") for name, states in by_state.items()
        },
        "sarima_orders": collected["orders"],
        "per_state": by_state,
    }
    write_rows(output / "predictions.csv", collected["rows"])
    write_json(output / "metrics.json", metrics)
    write_json(output / "selection.json", selection)
    print(f"Wrote {output / 'selection.json'}", flush=True)


if __name__ == "__main__":
    main()
