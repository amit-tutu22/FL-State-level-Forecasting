"""Generic sector forecast runner (Phase 4).

Same code path for every sector JSON under fl_experiment/sectors/.

Usage:
    python run_sector_forecast.py --sector public_finance_gst
    python run_sector_forecast.py --sector education_enrollment --rounds 5
    python run_sector_forecast.py --list
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from flwr.common import Metrics
from flwr.simulation import start_simulation

from metrics import regression_metrics
from sector_client import sector_client_fn, write_sector_cache
from sector_data import (
    build_forecast_frame,
    client_partitions,
    dataframe_to_xy,
    fit_feature_scaler,
    naive_seasonal_predict,
    predict_from_model_output,
    temporal_masks,
)
from sector_models import build_forecast_model
from task_spec import TaskSpec, list_sectors, load_task_spec

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SEED = 42


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total = sum(num for num, _ in metrics)
    if total == 0:
        return {}
    return {
        key: sum(num * m[key] for num, m in metrics if key in m) / total
        for key in metrics[0][1]
    }


def train_central(spec: TaskSpec, train_df, val_df, test_df, x_scaler) -> Dict[str, float]:
    X_train, y_train = dataframe_to_xy(spec, train_df, x_scaler)
    X_val, y_val = dataframe_to_xy(spec, val_df, x_scaler)
    X_test, y_test = dataframe_to_xy(spec, test_df, x_scaler)
    model = build_forecast_model(len(spec.feature_cols))
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    best, best_state, bad = float("inf"), None, 0

    for _ in range(400):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(X_val), y_val).item()
        if v < best - 1e-9:
            best = v
            best_state = {k: p.detach().cpu().clone() for k, p in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= 30:
                break
    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred = predict_from_model_output(spec, model(X_test), test_df)
    return regression_metrics(test_df["target"].values, pred)


def run_federated(spec: TaskSpec, parts, test_df, x_scaler, rounds: int, local_epochs: int):
    write_sector_cache(spec, parts, test_df, x_scaler, local_epochs)
    num_clients = len(parts)
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=min(5, num_clients),
        min_available_clients=num_clients,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
    )
    hist = start_simulation(
        client_fn=sector_client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )
    out = {}
    for key in ("rmse", "mae", "mape"):
        series = hist.metrics_distributed.get(key, [])
        out[key] = float(series[-1][1]) if series else float("nan")
    return out


def run_sector(sector_id: str, rounds: int, local_epochs: int) -> pd.DataFrame:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    spec = load_task_spec(sector_id)
    if spec.task_type != "forecast":
        raise SystemExit(f"Sector '{sector_id}' task_type={spec.task_type}; this runner is forecast-only.")

    frame = build_forecast_frame(spec)
    train_m, val_m, test_m = temporal_masks(spec, frame)
    train_df = frame.loc[train_m].reset_index(drop=True)
    val_df = frame.loc[val_m].reset_index(drop=True)
    test_df = frame.loc[test_m].reset_index(drop=True)
    x_scaler = fit_feature_scaler(spec, train_df)
    parts = client_partitions(spec, train_df)

    print(f"\n=== Sector: {spec.display_name} ({spec.sector_id}) ===")
    print(f"Features: {spec.feature_cols}")
    print(f"Rows train={len(train_df)} val={len(val_df)} test={len(test_df)} clients={len(parts)}")

    naive = regression_metrics(
        test_df["target"].values, naive_seasonal_predict(spec, test_df)
    )
    central = train_central(spec, train_df, val_df, test_df, x_scaler)
    federated = run_federated(spec, parts, test_df, x_scaler, rounds, local_epochs)

    rows = [
        {"sector_id": sector_id, "method": "naive_seasonal", **naive},
        {"sector_id": sector_id, "method": "centralized", **central},
        {"sector_id": sector_id, "method": "federated_fedavg", **federated},
    ]
    for row in rows:
        print(
            f"  {row['method']:<18} RMSE={row['rmse']:.2f}  "
            f"MAE={row['mae']:.2f}  MAPE%={row['mape']:.2f}"
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Phase 4 sector-agnostic forecast runner")
    parser.add_argument("--sector", type=str, default=None, help="Sector id (JSON stem)")
    parser.add_argument("--all", action="store_true", help="Run all forecast sectors")
    parser.add_argument("--list", action="store_true", help="List available sectors")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=5)
    args = parser.parse_args()

    if args.list:
        for s in list_sectors():
            spec = load_task_spec(s)
            print(f"{s}: {spec.display_name} ({spec.task_type})")
        return

    sectors = list_sectors() if args.all else [args.sector or "education_enrollment"]
    all_rows = []
    for sector_id in sectors:
        all_rows.append(run_sector(sector_id, args.rounds, args.local_epochs))

    results = pd.concat(all_rows, ignore_index=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "phase4_sectors.csv"
    results.to_csv(out, index=False)
    results.to_json(RESULTS_DIR / "phase4_sectors.json", orient="records", indent=2)
    print(f"\nWrote {out}")

    # Gate: for each sector, centralized or federated should beat naive on RMSE.
    ok = True
    for sector_id, grp in results.groupby("sector_id"):
        naive_rmse = float(grp.loc[grp.method == "naive_seasonal", "rmse"].iloc[0])
        best = float(grp.loc[grp.method != "naive_seasonal", "rmse"].min())
        beats = best < naive_rmse
        print(f"Gate {sector_id}: best RMSE {best:.2f} vs naive {naive_rmse:.2f} -> {'PASS' if beats else 'FAIL'}")
        ok = ok and beats
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
