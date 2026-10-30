"""FH-PFL v3 and its comparators B0-B5 on the frozen rolling origins.

    python -m fl_experiment.run_residual --split val \
        --selection docs/experiments/2026-10-29-selection.json \
        --source panel --allow-assumed-vintages \
        --panel data/reports/lagged45-2012/panel.csv

Settings come from the frozen selection file. v3 adds no searched
hyperparameter. `--engine flower` runs v3's full-correction model (B5)
through Flower clients; everything else is in-process. Scoring the test year
is a second look and needs `--second-look`. See docs/methods/fh_pfl_v3.md.
"""

from collections import Counter
from pathlib import Path

import numpy as np

from fl_experiment.baselines import GRIDS
from fl_experiment.reporting import model_summaries, summarize_rows
from fl_experiment.residual import (
    INPUTS,
    MODEL_NAMES,
    SARIMA_FALLBACK,
    TRAINED,
    federated_residuals,
    frozen_apfl,
    frozen_orders,
    local_residuals,
    residual_clients,
    residual_columns,
    ridge_residuals,
    sarima_bases,
    sarima_only_rows,
    shrunk_rows,
)
from fl_experiment.rolling import origin_months, states_for_month
from fl_experiment.runner import base_record, finish_run, seed_plans, start_run, variant_arguments

RESULTS = Path(__file__).resolve().parent / "results"


def prepare(folds, split, grid, maxiter, limit=None):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Origins, SARIMA bases and residual clients, shared by every variant and seed."""
    protocol = folds["protocol"]
    months = origin_months(folds, split)[:limit]
    groups = {month: states_for_month(folds, month, split) for month in months}
    orders = frozen_orders(groups, protocol, grid, maxiter)
    bases = {
        month: sarima_bases(group, orders, protocol, maxiter)
        for month, group in groups.items()
    }
    clients = {
        inputs: {
            month: residual_clients(
                group, bases[month], residual_columns(folds["feature_names"], inputs), protocol
            )
            for month, group in groups.items()
        }
        for inputs in INPUTS
    }
    return {"groups": groups, "orders": orders, "bases": bases, "clients": clients}


def _split(by_state):
    present = [client for client in by_state.values() if client is not None]
    missing = {state for state, client in by_state.items() if client is None}
    return present, missing


def deterministic_rows(prepared, seeds):
    """B0 (SARIMA) and B3 (ridge), repeated under every seed so seed counts line up."""
    rows = []
    for month, group in prepared["groups"].items():
        bases = prepared["bases"][month]
        batch = [{
            "model": MODEL_NAMES["b0"],
            "state_id": state_id,
            "target_month": month,
            "actual": float(origin["actual"]),
            "predicted": bases[state_id]["forecast"],
            "naive": origin["seasonal_naive"],
            "fallback": SARIMA_FALLBACK if bases[state_id]["fallback"] else "",
        } for state_id, _, origin in group]
        present, missing = _split(prepared["clients"]["gst_ewb"][month])
        ridge, _ = shrunk_rows(
            MODEL_NAMES["b3"], present, [ridge_residuals(client) for client in present]
        )
        batch += ridge + sarima_only_rows(MODEL_NAMES["b3"], group, bases, missing)
        for seed in seeds:
            rows.extend(dict(row, seed=seed) for row in batch)
    return rows


def _flower_rows(clients, settings, job, cache):
    """B5 through Flower. Imported here so the in-process run does not start Ray."""
    from fl_experiment.flower_rolling import (  # pylint: disable=import-outside-toplevel
        flower_origin,
    )
    rows, info = flower_origin(clients, settings, job, cache)
    for client, row in zip(clients, rows):
        row["model"] = MODEL_NAMES["b5"]
        row["fallback"] = SARIMA_FALLBACK if client["sarima_fallback"] else ""
    return rows, info


def _origin_rows(plan, month, present, prepared):
    """One trained variant at one origin month: rows and training info."""
    kind, settings = plan["kind"], plan["settings"]
    trainer = TRAINED[kind]["trainer"]
    if plan["engine"] == "flower" and kind == "v3":
        cache = plan["cache"] / f"v3-{settings['seed']}-{month}"
        return _flower_rows(present, settings, plan["jobs"]["apfl"], cache)
    if trainer == "local":
        residuals, info = local_residuals(present, settings, prepared["state_seeds"]), {}
    else:
        residuals, info = federated_residuals(present, settings, plan["jobs"][trainer])
    rows, info["lambdas"] = shrunk_rows(MODEL_NAMES[kind], present, residuals)
    if kind == "v3":
        rows += shrunk_rows(MODEL_NAMES["b5"], present, residuals, fixed=1.0)[0]
    return rows, info


def variant_rows(plan, prepared):
    """Every origin of one trained variant and seed."""
    rows, infos = [], {}
    inputs = TRAINED[plan["kind"]]["inputs"]
    for month, group in prepared["groups"].items():
        present, missing = _split(prepared["clients"][inputs][month])
        batch, infos[month] = _origin_rows(plan, month, present, prepared)
        names = {row["model"] for row in batch}
        for name in sorted(names):
            batch += sarima_only_rows(name, group, prepared["bases"][month], missing)
        rows.extend(dict(row, seed=plan["settings"]["seed"]) for row in batch)
    return rows, infos


def decision(runs):
    """The pre-registered checks: v3 against every comparator, mean and per seed."""
    def by_seed(model):
        return {seed: aggregate["rmse_equal_state"] for seed, aggregate, _ in runs[model]}

    v3 = by_seed(MODEL_NAMES["v3"])
    report = {}
    for key, name in MODEL_NAMES.items():
        if key == "v3" or name not in runs:
            continue
        rival = by_seed(name)
        wins = sum(v3[seed] < rival[seed] for seed in v3)
        gap = float(np.mean(list(v3.values())) - np.mean(list(rival.values())))
        report[name] = {
            "v3_minus_rival_mean": gap,
            "seeds_v3_lower": int(wins),
            "seeds": len(v3),
            "v3_lower": bool(gap < 0 and wins >= 2),
        }
    return report


def lambda_counts(infos):
    """How often each shrinkage value was chosen, per trained variant."""
    counts = {}
    for key, months in infos.items():
        tally = Counter()
        for info in months.values():
            tally.update(str(value) for value in info.get("lambdas", {}).values())
        counts[key] = dict(sorted(tally.items()))
    return counts


def _run(args, run):
    folds = run["folds"]
    prepared = prepare(folds, args.split, GRIDS[args.order_grid], args.maxiter, args.limit_origins)
    prepared["state_seeds"] = {state: index for index, state in enumerate(folds["states"])}
    groups = prepared["groups"]
    print(f"{len(groups)} origins, {len(next(iter(groups.values())))} states", flush=True)
    rows = deterministic_rows(prepared, args.seeds or run["selection"]["seeds"])
    infos = {}
    for kind, plan in seed_plans(args, run, MODEL_NAMES):
        plan["jobs"] = frozen_apfl(run["selection"])
        batch, infos[f"{kind}-{plan['settings']['seed']}"] = variant_rows(plan, prepared)
        rows.extend(batch)
    missing = {
        inputs: sum(client is None for month in by_month.values() for client in month.values())
        for inputs, by_month in prepared["clients"].items()
    }
    return rows, {
        "orders": prepared["orders"],
        "origins_without_residual_fit": missing,
        "sarima_fallbacks": sum(
            base["fallback"] for month in prepared["bases"].values() for base in month.values()
        ),
        "lambda_counts": lambda_counts(infos),
        "rounds": {
            key: {month: {k: v for k, v in info.items() if k != "lambdas"}
                  for month, info in months.items()}
            for key, months in infos.items()
        },
    }


def main():
    """Run FH-PFL v3 and its comparators with the frozen settings, and write the record."""
    args = variant_arguments(__doc__, TRAINED)
    run = start_run(args, RESULTS / f"fh-pfl-v3-{args.split}")
    rows, extra = _run(args, run)
    record = {**base_record(args, run), **model_summaries(rows), **extra}
    runs = summarize_rows(rows)
    if MODEL_NAMES["v3"] in runs:
        record["decision"] = decision(runs)
    finish_run(run, rows, record)


if __name__ == "__main__":
    main()
