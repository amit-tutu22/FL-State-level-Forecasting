"""Describe each state's GST series and relate it to which method wins there (#20).

    python -m fl_experiment.heterogeneity \
        --panel data/reports/lagged45-2012/panel.csv \
        --validation fl_experiment/results/real-2026-10-29/protocol/metrics.json \
        --test fl_experiment/results/real-2026-10-29/protocol-test/metrics.json

Descriptors use target months before the first validation month only.
Correlations are Spearman over the states in both runs. With about 32
states they are descriptive, and no significance test is reported.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from data_pipeline.clients import load_clients
from fl_experiment.state_data import PANEL_TARGET, shift_month

NEURAL = ("local_gru_gst_only", "pooled_gru", "fedavg", "fedprox_mu_0.01", "apfl_step_1")
DESCRIPTORS = ("mean_level", "volatility", "seasonal_strength", "history_months", "dip_2020")


def target_series(panel, cutoff):
    """Observed target by state and month, for months before `cutoff`."""
    series = {}
    for state, rows in load_clients(panel, allow_assumed_vintages=True).items():
        values = {
            row["target_month"]: float(row[PANEL_TARGET])
            for row in rows
            if row[PANEL_TARGET] != "" and row["target_month"] < cutoff
        }
        if values:
            series[state] = values
    return series


def _seasonal_strength(months, logs):
    """Share of detrended log variance explained by month-of-year means."""
    index = np.arange(len(logs), dtype=float)
    trend = np.polyval(np.polyfit(index, logs, 1), index)
    residual = logs - trend
    calendar = np.asarray([int(month[5:7]) for month in months])
    seasonal = np.zeros_like(residual)
    for month in set(calendar):
        mask = calendar == month
        seasonal[mask] = residual[mask].mean()
    total = float(np.var(residual))
    return float(np.var(seasonal) / total) if total > 0 else 0.0


def _dip_2020(values):
    """One minus April–June 2020 over April–June 2019. Missing months give NaN."""
    spring = [values.get(f"{year}-{month:02d}") for year in (2019, 2020) for month in (4, 5, 6)]
    if any(value is None for value in spring):
        return float("nan")
    return float(1.0 - sum(spring[3:]) / sum(spring[:3]))


def describe(values):
    """Level, volatility, seasonality, history and the 2020 dip for one state."""
    months = sorted(values)
    logs = np.log1p(np.asarray([values[month] for month in months]))
    changes = [
        np.log1p(values[month]) - np.log1p(values[shift_month(month, -1)])
        for month in months
        if shift_month(month, -1) in values
    ]
    return {
        "mean_level": float(np.mean([values[month] for month in months])),
        "volatility": float(np.std(changes)) if len(changes) > 1 else float("nan"),
        "seasonal_strength": _seasonal_strength(months, logs),
        "history_months": len(months),
        "dip_2020": _dip_2020(values),
    }


def state_rmse(metrics, model):
    """Mean-across-seeds RMSE by state for one model in a run record."""
    return {state: item["rmse"]["mean"] for state, item in metrics["per_state"][model].items()}


def outcomes(metrics):
    """Relative differences that say where personalization and federation help."""
    rmse = {model: state_rmse(metrics, model) for model in NEURAL + ("sarima",)}
    result = {}
    for state, local in rmse["local_gru_gst_only"].items():
        best_neural = min(rmse[model][state] for model in NEURAL)
        result[state] = {
            "apfl_vs_local": rmse["apfl_step_1"][state] / local - 1.0,
            "fedavg_vs_local": rmse["fedavg"][state] / local - 1.0,
            "sarima_vs_best_neural": rmse["sarima"][state] / best_neural - 1.0,
        }
    return result


def correlations(descriptors, results):
    """Spearman rank correlation for each descriptor and outcome pair."""
    table = {}
    states = sorted(set(descriptors) & set(results))
    for name in DESCRIPTORS:
        table[name] = {}
        for outcome in ("apfl_vs_local", "fedavg_vs_local", "sarima_vs_best_neural"):
            pairs = [
                (descriptors[state][name], results[state][outcome])
                for state in states
                if np.isfinite(descriptors[state][name])
            ]
            rho = spearmanr([a for a, _ in pairs], [b for _, b in pairs]).statistic
            table[name][outcome] = {"rho": float(rho), "states": len(pairs)}
    return table


def parse_args():
    """Parse the input paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--cutoff", default="2024-04", help="first validation month")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    """Write descriptors, outcomes and correlations for both evaluation years."""
    args = parse_args()
    series = target_series(args.panel, args.cutoff)
    descriptors = {state: describe(values) for state, values in series.items()}
    report = {"cutoff": args.cutoff, "descriptors": {}, "outcomes": {}, "spearman": {}}
    for label, path in (("validation", args.validation), ("test", args.test)):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        results = outcomes(metrics)
        report["outcomes"][label] = results
        report["spearman"][label] = correlations(descriptors, results)
    scored = set(report["outcomes"]["validation"])
    report["descriptors"] = {state: descriptors[state] for state in sorted(scored)}
    output = args.output or Path(__file__).resolve().parent / "results" / "heterogeneity.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for label, table in report["spearman"].items():
        print(label)
        for name, row in table.items():
            cells = ", ".join(f"{key} {value['rho']:+.2f}" for key, value in row.items())
            print(f"  {name}: {cells}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
