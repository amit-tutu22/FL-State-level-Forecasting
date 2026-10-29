"""FedAvg, FedProx and APFL through Flower clients on the frozen rolling origins.

    python -m fl_experiment.flower_rolling --split val \
        --selection docs/experiments/2026-10-29-selection.json \
        --reference fl_experiment/results/real-2026-10-29/protocol/metrics.json \
        --source panel --allow-assumed-vintages \
        --panel data/reports/lagged45-2012/panel.csv

Settings come from the frozen selection file and are not searched again. Each
origin is one Flower simulation over its eligible states. The client update
is the same code the in-process runs use (`federated.py`). Scoring the test
year here is a second look and needs `--second-look`.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import flwr as fl
import torch
from flwr.simulation import start_simulation

from fl_experiment.client import make_origin_client_fn, write_origin_cache
from fl_experiment.evaluate_test import require_selection, settings_from
from fl_experiment.federated import initial_parameters, make_job, origin_row
from fl_experiment.folds import build_folds, protocol_hash
from fl_experiment.local_gru import compute_info
from fl_experiment.reporting import collate, ranked_states, write_json, write_rows
from fl_experiment.baselines import summarize_model
from fl_experiment.rolling import federated_clients, origin_months, states_for_month
from fl_experiment.server import OriginRecorder, build_origin_strategy
from fl_experiment.state_data import add_data_arguments, records_from_arguments

RESULTS = Path(__file__).resolve().parent / "results"
RAY_ARGS = {"include_dashboard": False, "ignore_reinit_error": True, "log_to_driver": False}


def flower_origin(clients, settings, job, cache):
    """Run one Flower simulation at one origin and return its prediction rows."""
    write_origin_cache(clients, cache, settings, job)
    n_features = int(clients[0]["fit"][0].shape[-1])
    recorder = OriginRecorder(settings["round_patience"], job["personalize"])
    strategy = build_origin_strategy(
        job, len(clients), recorder, initial_parameters(n_features, settings)
    )
    start_simulation(
        client_fn=make_origin_client_fn(cache),
        num_clients=len(clients),
        config=fl.server.ServerConfig(num_rounds=settings["max_rounds"]),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args=RAY_ARGS,
        keep_initialised=True,
    )
    predictions = recorder.best["predictions"]
    rows = [origin_row(client, predictions[client["state_id"]]) for client in clients]
    info = {
        "selected_round": recorder.best["round"],
        "rounds_run": len(recorder.history),
        "alphas": recorder.best["alphas"],
    }
    return rows, info


def model_name(kind, selection):
    """The same model names the in-process comparison uses."""
    if kind == "fedprox":
        return f"fedprox_mu_{selection['fedprox_mu']:g}"
    if kind == "apfl":
        return f"apfl_step_{selection['apfl_alpha_step']:g}"
    return "fedavg"


def frozen_job(kind, selection):
    """Method settings read from the frozen selection file."""
    value = {"fedprox": selection["fedprox_mu"], "apfl": selection["apfl_alpha_step"]}
    return make_job(kind, value.get(kind, 0.0), selection["apfl_alpha_init"])


def flower_rows(folds, selection, plan, cache_root):
    """Every eligible origin of `plan['split']` for one method and seed."""
    job = frozen_job(plan["kind"], selection)
    settings = settings_from(selection, plan["seed"])
    rows, rounds = [], {}
    for month in origin_months(folds, plan["split"])[: plan["limit_origins"]]:
        group = states_for_month(folds, month, plan["split"])[: plan["limit_states"]]
        clients = federated_clients(group, selection["feature_set"], folds)
        cache = cache_root / f"{plan['kind']}-{plan['seed']}-{month}"
        batch, rounds[month] = flower_origin(clients, settings, job, cache)
        for row in batch:
            row["model"] = model_name(plan["kind"], selection)
            row["seed"] = plan["seed"]
        rows.extend(batch)
    return rows, rounds


def agreement(summaries, reference):
    """Flower against in-process equal-state RMSE, with the seed spread as tolerance."""
    report = {}
    for name, summary in summaries.items():
        if name not in reference:
            continue
        flower = summary["rmse_equal_state"]
        local = reference[name]["rmse_equal_state"]
        tolerance = max(flower["std"], local["std"])
        gap = flower["mean"] - local["mean"]
        report[name] = {
            "flower_mean": flower["mean"],
            "flower_std": flower["std"],
            "in_process_mean": local["mean"],
            "in_process_std": local["std"],
            "gap": gap,
            "within_seed_spread": abs(gap) <= tolerance,
        }
    return report


def parse_args():
    """Parse the Flower rolling-origin options. There is no hyperparameter grid."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--second-look", action="store_true",
                        help="required for --split test: the test year was already scored")
    parser.add_argument("--methods", nargs="+", default=["fedavg", "fedprox", "apfl"],
                        choices=("fedavg", "fedprox", "apfl"))
    parser.add_argument("--reference", type=Path, help="in-process metrics.json to compare")
    parser.add_argument("--limit-origins", type=int, help="smoke tests only")
    parser.add_argument("--limit-states", type=int, help="smoke tests only")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _run(args, folds, selection, output):
    rows, rounds = [], {}
    for kind in args.methods:
        print(kind, flush=True)
        for seed in selection["seeds"]:
            print(f"  seed {seed}", flush=True)
            plan = {
                "kind": kind, "seed": seed, "split": args.split,
                "limit_origins": args.limit_origins, "limit_states": args.limit_states,
            }
            batch, info = flower_rows(folds, selection, plan, output / "cache")
            rows.extend(batch)
            rounds[f"{kind}-{seed}"] = info
    return rows, rounds


def _summaries(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["model"], row["seed"]), []).append(row)
    runs = {}
    for (model, _), items in sorted(grouped.items()):
        runs.setdefault(model, []).append(summarize_model(items))
    summaries, by_state = {}, {}
    for model, pairs in runs.items():
        summaries[model], by_state[model] = collate(pairs)
    return summaries, by_state


def main():
    """Run the frozen federated methods through Flower and write the record."""
    args = parse_args()
    if args.split == "test" and not args.second_look:
        raise SystemExit("The test year was already scored. Pass --second-look to re-score it.")
    if not args.selection.is_file():
        raise SystemExit(f"selection file not found: {args.selection}")
    logging.getLogger("flwr").setLevel(logging.WARNING)
    torch.set_num_threads(1)
    started = time.perf_counter()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    records, feature_names, val_start, test_start = records_from_arguments(args)
    folds = build_folds(records, feature_names, val_start, test_start, args.min_train, args.window)
    require_selection(selection, folds)
    output = args.output_dir or RESULTS / f"flower-{args.split}"
    output.mkdir(parents=True, exist_ok=True)
    rows, rounds = _run(args, folds, selection, output)
    record = _record(args, folds, selection, rows, rounds)
    record["compute"] = compute_info(started)
    write_rows(output / "predictions.csv", rows)
    write_json(output / "metrics.json", record)
    print(f"Wrote {output / 'metrics.json'}", flush=True)
    if not all(item["within_seed_spread"] for item in record.get("agreement", {}).values()):
        raise SystemExit("Flower and in-process results differ by more than the seed spread")


def _record(args, folds, selection, rows, rounds):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Run record, plus the agreement table when an in-process reference is given."""
    summaries, by_state = _summaries(rows)
    record = {
        "split": args.split,
        "second_look": args.split == "test",
        "protocol_hash": protocol_hash(folds),
        "selection": selection,
        "summaries": summaries,
        "largest_rmse": {name: ranked_states(states, "rmse") for name, states in by_state.items()},
        "per_state": by_state,
        "rounds": rounds,
    }
    if args.reference:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))["summaries"]
        record["agreement"] = agreement(summaries, reference)
        for name, item in record["agreement"].items():
            print(
                f"{name}: Flower {item['flower_mean']:.2f} ± {item['flower_std']:.2f}, "
                f"in-process {item['in_process_mean']:.2f} ± {item['in_process_std']:.2f}",
                flush=True,
            )
    return record


if __name__ == "__main__":
    main()
