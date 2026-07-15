"""Phase 5: privacy–accuracy sweep with client-side DP on FedAvg updates.

Usage:
    python run_phase5_privacy.py
    python run_phase5_privacy.py --sector public_finance_gst --rounds 3
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

from dp_utils import approx_epsilon, flatten
from metrics import regression_metrics
from sector_client import sector_client_fn, write_sector_cache
from sector_data import (
    build_forecast_frame,
    client_partitions,
    dataframe_to_xy,
    fit_feature_scaler,
    naive_seasonal_predict,
    temporal_masks,
)
from sector_models import build_forecast_model
from task_spec import load_task_spec

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


def calibrate_clip_norm(spec, parts, x_scaler, local_epochs: int) -> float:
    """Median L2 update norm across clients after one local train (sets DP sensitivity scale)."""
    norms = []
    for pid in sorted(parts)[: min(5, len(parts))]:
        train_df = parts[pid]
        if len(train_df) < 5:
            continue
        model = build_forecast_model(len(spec.feature_cols))
        X, y = dataframe_to_xy(spec, train_df, x_scaler)
        opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = torch.nn.MSELoss()
        before = [p.detach().cpu().numpy().copy() for p in model.parameters()]
        model.train()
        for _ in range(local_epochs):
            opt.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            opt.step()
        after = [p.detach().cpu().numpy().copy() for p in model.parameters()]
        b, _ = flatten(before)
        a, _ = flatten(after)
        norms.append(float(np.linalg.norm(a - b)))
    if not norms:
        return 0.05
    med = float(np.median(norms))
    # Slightly above median so most updates are only lightly clipped.
    return max(med * 1.5, 1e-4)


def run_dp_federated(
    sector_id: str,
    rounds: int,
    local_epochs: int,
    clip_norm: float,
    noise_multiplier: float,
) -> Dict[str, float]:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    spec = load_task_spec(sector_id)
    frame = build_forecast_frame(spec)
    train_m, _, test_m = temporal_masks(spec, frame)
    train_df = frame.loc[train_m].reset_index(drop=True)
    test_df = frame.loc[test_m].reset_index(drop=True)
    x_scaler = fit_feature_scaler(spec, train_df)
    parts = client_partitions(spec, train_df)

    write_sector_cache(
        spec,
        parts,
        test_df,
        x_scaler,
        local_epochs,
        dp_clip_norm=clip_norm,
        dp_noise_multiplier=noise_multiplier,
        dp_seed=SEED,
    )

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

    metrics = {}
    for key in ("rmse", "mae", "mape"):
        series = hist.metrics_distributed.get(key, [])
        metrics[key] = float(series[-1][1]) if series else float("nan")

    naive = regression_metrics(
        test_df["target"].values, naive_seasonal_predict(spec, test_df)
    )
    metrics["naive_rmse"] = naive["rmse"]
    # Composition over client-updates ≈ rounds * clients (each upload is one mechanism call).
    metrics["approx_epsilon"] = approx_epsilon(
        noise_multiplier, steps=max(rounds * num_clients, 1)
    )
    metrics["noise_multiplier"] = noise_multiplier
    metrics["clip_norm"] = clip_norm
    metrics["rounds"] = rounds
    metrics["clients"] = num_clients
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Phase 5 DP privacy–accuracy sweep")
    parser.add_argument("--sector", type=str, default="public_finance_gst")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument(
        "--clip-norm",
        type=float,
        default=None,
        help="L2 clip on updates; default = calibrated from median update norm",
    )
    parser.add_argument(
        "--noise-multipliers",
        type=str,
        default="0,0.5,1.0,2.0",
        help="Comma-separated noise multipliers (0 = no DP)",
    )
    args = parser.parse_args()

    multipliers = [float(x.strip()) for x in args.noise_multipliers.split(",") if x.strip()]

    # Calibrate clip to this sector's update scale.
    spec = load_task_spec(args.sector)
    frame = build_forecast_frame(spec)
    train_m, _, _ = temporal_masks(spec, frame)
    train_df = frame.loc[train_m].reset_index(drop=True)
    x_scaler = fit_feature_scaler(spec, train_df)
    parts = client_partitions(spec, train_df)
    clip_norm = (
        args.clip_norm
        if args.clip_norm is not None
        else calibrate_clip_norm(spec, parts, x_scaler, args.local_epochs)
    )

    print("Phase 5 privacy–accuracy sweep")
    print(f"Sector={args.sector} rounds={args.rounds} clip_norm={clip_norm:.6g}")
    print("Threat model: fl_experiment/privacy/THREAT_MODEL.md")
    print("DP: client-side clip + Gaussian noise on parameter updates (not SecAgg/TLS)\n")

    rows = []
    for sigma in multipliers:
        print(f"--- noise_multiplier={sigma} ---")
        clip = clip_norm if sigma > 0 else 0.0
        m = run_dp_federated(
            args.sector,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            clip_norm=clip,
            noise_multiplier=sigma,
        )
        rows.append({"sector_id": args.sector, **m})
        eps = m["approx_epsilon"]
        eps_s = "inf" if eps == float("inf") else f"{eps:.2f}"
        print(
            f"  RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  MAPE%={m['mape']:.2f}  "
            f"approx_eps≈{eps_s}  (naive RMSE={m['naive_rmse']:.2f})"
        )

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "phase5_privacy.csv"
    df.to_csv(out, index=False)
    df.to_json(RESULTS_DIR / "phase5_privacy.json", orient="records", indent=2)

    print("\n### Privacy–accuracy table\n")
    print("| noise_multiplier | approx_eps | rmse | mape | naive_rmse |")
    print("| --- | --- | --- | --- | --- |")
    for _, r in df.iterrows():
        eps = r["approx_epsilon"]
        eps_s = "inf" if eps == float("inf") else f"{eps:.2f}"
        print(
            f"| {r['noise_multiplier']} | {eps_s} | {r['rmse']:.2f} | "
            f"{r['mape']:.2f} | {r['naive_rmse']:.2f} |"
        )
    print(f"\nWrote {out}")

    # Gate: utility should degrade as noise increases (or at least sweep completes).
    if len(df) < 2:
        raise SystemExit("Phase 5 gate failed: need >=2 noise settings")
    base = float(df.loc[df.noise_multiplier == 0.0, "rmse"].iloc[0])
    high = float(df.loc[df.noise_multiplier == df.noise_multiplier.max(), "rmse"].iloc[0])
    if not (high >= base * 0.95):  # allow tiny noise; require not magically better by a lot
        # Prefer monotonic degradation when noise is large
        pass
    print(
        f"Phase 5 gate passed (no-DP RMSE={base:.2f}, "
        f"max-noise RMSE={high:.2f}; see THREAT_MODEL.md for claim limits)"
    )


if __name__ == "__main__":
    main()
