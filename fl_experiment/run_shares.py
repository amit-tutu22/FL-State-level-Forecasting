"""FH-PFL v2 and its ablations on the frozen rolling origins.

    python -m fl_experiment.run_shares --split val \
        --selection docs/experiments/2026-10-29-selection.json \
        --source panel --allow-assumed-vintages \
        --panel data/reports/lagged45-2012/panel.csv

Settings come from the frozen selection file. FH-PFL v2 adds no
hyperparameter. `--engine flower` runs v2 itself through Flower clients;
the ablations are always in-process. Scoring the test year is a second look
and needs `--second-look`. See docs/methods/fh_pfl_v2.md.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from fl_experiment.baselines import GRIDS
from fl_experiment.evaluate_test import add_second_look_arguments, frozen_folds, settings_from
from fl_experiment.folds import protocol_hash
from fl_experiment.local_gru import compute_info
from fl_experiment.reporting import model_summaries, summarize_rows, write_json, write_rows
from fl_experiment.rolling import origin_months, states_for_month
from fl_experiment.shares import (
    ANCHOR_FALLBACK,
    MODEL_NAMES,
    TRAINED,
    national_anchors,
    run_share_origin,
    same_federation,
    share_clients,
    top_down_rows,
    trained_rows,
)
from fl_experiment.state_data import add_data_arguments

RESULTS = Path(__file__).resolve().parent / "results"
RIVALS = ("a1", "a2")


def parse_args():
    """Parse the FH-PFL v2 options. There is no hyperparameter grid."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    add_second_look_arguments(parser)
    parser.add_argument("--variants", nargs="+", default=list(TRAINED), choices=list(TRAINED))
    parser.add_argument("--engine", choices=("in_process", "flower"), default="in_process")
    parser.add_argument("--order-grid", choices=sorted(GRIDS), default="full")
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument("--seeds", type=int, nargs="+", help="default: the frozen seeds")
    parser.add_argument("--limit-origins", type=int, help="smoke tests only")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _flower_rows(clients, settings, anchor, cache):
    """v2 through Flower. Imported here so the in-process run does not start Ray."""
    from fl_experiment.shares_flower import (  # pylint: disable=import-outside-toplevel
        flower_share_origin,
    )
    shares, info = flower_share_origin(clients, settings, cache)
    rows = []
    for client in clients:
        rows.append({
            "model": MODEL_NAMES["v2"],
            "state_id": client["state_id"],
            "target_month": client["target_month"],
            "actual": client["actual"],
            "predicted": shares[client["state_id"]] * anchor["sarima"],
            "naive": client["naive"],
            "fallback": ANCHOR_FALLBACK if anchor["fallback"] else "",
        })
    return rows, info


def variant_rows(plan, groups, anchors, folds):
    """Every origin of one trained variant and seed."""
    rows, rounds = [], {}
    switches = TRAINED[plan["kind"]]
    for month, group in groups.items():
        clients = share_clients(group, folds, switches["national_input"])
        if plan["engine"] == "flower" and plan["kind"] == "v2":
            cache = plan["cache"] / f"v2-{plan['settings']['seed']}-{month}"
            batch, rounds[month] = _flower_rows(clients, plan["settings"], anchors[month], cache)
        else:
            best = run_share_origin(clients, plan["settings"], switches["coupled"])
            batch = trained_rows(plan["kind"], clients, best["origin_scores"], anchors[month])
            rounds[month] = {"selected_round": best["round"], "rounds_run": best["rounds_run"]}
        for row in batch:
            row["seed"] = plan["settings"]["seed"]
        rows.extend(batch)
    return rows, rounds


def top_down(groups, anchors, folds, seeds):
    """A3 is deterministic; it is repeated under every seed so seed counts line up."""
    rows = []
    for month, group in groups.items():
        batch = top_down_rows(share_clients(group, folds), anchors[month])
        for seed in seeds:
            rows.extend(dict(row, seed=seed) for row in batch)
    return rows


def decision(runs):
    """The pre-registered federation check: v2 against A1 and A2, mean and per seed."""
    def by_seed(model):
        return {seed: aggregate["rmse_equal_state"] for seed, aggregate, _ in runs[model]}

    v2 = by_seed(MODEL_NAMES["v2"])
    report = {}
    for key in RIVALS:
        rival = by_seed(MODEL_NAMES[key])
        wins = sum(v2[seed] < rival[seed] for seed in v2)
        mean_gap = float(np.mean(list(v2.values())) - np.mean(list(rival.values())))
        report[MODEL_NAMES[key]] = {
            "v2_minus_rival_mean": mean_gap,
            "seeds_v2_lower": int(wins),
            "seeds": len(v2),
            "v2_lower": bool(mean_gap < 0 and wins >= 2),
        }
    report["federation_claim_on_this_split"] = all(
        item["v2_lower"] for item in report.values()
    )
    return report


def _groups(folds, split, limit):
    months = origin_months(folds, split)[:limit]
    groups = {month: states_for_month(folds, month, split) for month in months}
    same_federation(groups)
    return groups


def _run(args, folds, selection, output):
    groups = _groups(folds, args.split, args.limit_origins)
    protocol = folds["protocol"]
    print(f"{len(groups)} origins, {len(next(iter(groups.values())))} states", flush=True)
    anchors, order = national_anchors(
        groups, protocol["val_start"], GRIDS[args.order_grid], protocol["min_train"],
        args.maxiter,
    )
    print(f"national SARIMA order {order['order']} {order['seasonal_order']}", flush=True)
    seeds = args.seeds or selection["seeds"]
    rows, rounds = top_down(groups, anchors, folds, seeds), {}
    for kind in args.variants:
        for seed in seeds:
            print(f"{MODEL_NAMES[kind]} seed {seed}", flush=True)
            plan = {
                "kind": kind, "engine": args.engine, "cache": output / "cache",
                "settings": settings_from(selection, seed),
            }
            batch, rounds[f"{kind}-{seed}"] = variant_rows(plan, groups, anchors, folds)
            rows.extend(batch)
    return rows, {"rounds": rounds, "anchors": anchors, "national_order": order}


def _record(args, folds, selection, rows, extra):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    record = {
        "split": args.split,
        "second_look": args.split == "test",
        "engine": args.engine,
        "protocol_hash": protocol_hash(folds),
        "selection": selection,
        "anchor_fallbacks": sum(item["fallback"] for item in extra["anchors"].values()),
        **model_summaries(rows),
        **extra,
    }
    runs = summarize_rows(rows)
    if MODEL_NAMES["v2"] in runs and all(MODEL_NAMES[key] in runs for key in RIVALS):
        record["decision"] = decision(runs)
    return record


def main():
    """Run FH-PFL v2 and its ablations with the frozen settings, and write the record."""
    args = parse_args()
    started = time.perf_counter()
    selection, folds = frozen_folds(args)
    torch.set_num_threads(1)
    if args.engine == "flower":
        logging.getLogger("flwr").setLevel(logging.WARNING)
    output = args.output_dir or RESULTS / f"fh-pfl-v2-{args.split}"
    output.mkdir(parents=True, exist_ok=True)
    rows, extra = _run(args, folds, selection, output)
    record = _record(args, folds, selection, rows, extra)
    record["compute"] = compute_info(started)
    write_rows(output / "predictions.csv", rows)
    write_json(output / "metrics.json", record)
    for name, summary in sorted(record["summaries"].items()):
        rmse = summary["rmse_equal_state"]
        print(f"{name}: equal-state RMSE {rmse['mean']:.1f} ± {rmse['std']:.1f}", flush=True)
    if "decision" in record:
        print(json.dumps(record["decision"], indent=2), flush=True)
    print(f"Wrote {output / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
