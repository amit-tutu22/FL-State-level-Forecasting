"""Command-line scaffolding shared by the FH-PFL v2 and v3 runners."""

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from fl_experiment.baselines import GRIDS
from fl_experiment.evaluate_test import add_second_look_arguments, frozen_folds, settings_from
from fl_experiment.folds import protocol_hash
from fl_experiment.local_gru import compute_info
from fl_experiment.reporting import write_json, write_rows
from fl_experiment.state_data import add_data_arguments


def variant_arguments(doc, variants):
    """Parse the options every frozen-selection variant runner takes."""
    parser = argparse.ArgumentParser(description=doc)
    add_data_arguments(parser)
    add_second_look_arguments(parser)
    parser.add_argument("--variants", nargs="+", default=list(variants), choices=list(variants))
    parser.add_argument("--engine", choices=("in_process", "flower"), default="in_process")
    parser.add_argument("--order-grid", choices=sorted(GRIDS), default="full")
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", help="default: the frozen seeds")
    parser.add_argument("--limit-origins", type=int, help="smoke tests only")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def start_run(args, default_output):
    """Load the frozen selection and folds, and make the output directory."""
    started = time.perf_counter()
    selection, folds = frozen_folds(args)
    torch.set_num_threads(1)
    if args.engine == "flower":
        logging.getLogger("flwr").setLevel(logging.WARNING)
    output = args.output_dir or default_output
    output.mkdir(parents=True, exist_ok=True)
    return {"started": started, "selection": selection, "folds": folds, "output": output}


def seed_plans(args, run, names):
    """(kind, plan) for every variant and seed, in run order."""
    for kind in args.variants:
        for seed in args.seeds or run["selection"]["seeds"]:
            print(f"{names[kind]} seed {seed}", flush=True)
            yield kind, {
                "kind": kind, "engine": args.engine, "cache": run["output"] / "cache",
                "settings": settings_from(run["selection"], seed),
            }


def base_record(args, run):
    """Fields every variant record starts with."""
    return {
        "split": args.split,
        "second_look": args.split == "test",
        "engine": args.engine,
        "protocol_hash": protocol_hash(run["folds"]),
        "selection": run["selection"],
    }


def finish_run(run, rows, record):
    """Write the predictions and the record, and print the summaries."""
    record["compute"] = compute_info(run["started"])
    write_rows(run["output"] / "predictions.csv", rows)
    write_json(run["output"] / "metrics.json", record)
    for name, summary in sorted(record["summaries"].items()):
        rmse = summary["rmse_equal_state"]
        print(f"{name}: equal-state RMSE {rmse['mean']:.1f} ± {rmse['std']:.1f}", flush=True)
    if "decision" in record:
        print(json.dumps(record["decision"], indent=2), flush=True)
    print(f"Wrote {run['output'] / 'metrics.json'}", flush=True)
