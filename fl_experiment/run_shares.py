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

from pathlib import Path

import numpy as np

from fl_experiment.baselines import GRIDS
from fl_experiment.reporting import model_summaries, summarize_rows
from fl_experiment.rolling import origin_months, states_for_month
from fl_experiment.runner import base_record, finish_run, seed_plans, start_run, variant_arguments
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

RESULTS = Path(__file__).resolve().parent / "results"
RIVALS = ("a1", "a2")


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


def _run(args, run):
    folds = run["folds"]
    groups = _groups(folds, args.split, args.limit_origins)
    protocol = folds["protocol"]
    print(f"{len(groups)} origins, {len(next(iter(groups.values())))} states", flush=True)
    anchors, order = national_anchors(
        groups, protocol["val_start"], GRIDS[args.order_grid], protocol["min_train"],
        args.maxiter,
    )
    print(f"national SARIMA order {order['order']} {order['seasonal_order']}", flush=True)
    seeds = args.seeds or run["selection"]["seeds"]
    rows, rounds = top_down(groups, anchors, folds, seeds), {}
    for kind, plan in seed_plans(args, run, MODEL_NAMES):
        seed = plan["settings"]["seed"]
        batch, rounds[f"{kind}-{seed}"] = variant_rows(plan, groups, anchors, folds)
        rows.extend(batch)
    return rows, {"rounds": rounds, "anchors": anchors, "national_order": order}


def main():
    """Run FH-PFL v2 and its ablations with the frozen settings, and write the record."""
    args = variant_arguments(__doc__, TRAINED)
    run = start_run(args, RESULTS / f"fh-pfl-v2-{args.split}")
    rows, extra = _run(args, run)
    record = {
        **base_record(args, run),
        "anchor_fallbacks": sum(item["fallback"] for item in extra["anchors"].values()),
        **model_summaries(rows),
        **extra,
    }
    runs = summarize_rows(rows)
    if MODEL_NAMES["v2"] in runs and all(MODEL_NAMES[key] in runs for key in RIVALS):
        record["decision"] = decision(runs)
    finish_run(run, rows, record)


if __name__ == "__main__":
    main()
