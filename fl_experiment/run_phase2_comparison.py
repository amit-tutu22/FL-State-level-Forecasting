"""Phase 2: compare local-only, centralized, and FedAvg under IID / non-IID splits.

Usage:
    python run_phase2_comparison.py
    python run_phase2_comparison.py --protocols by_state,iid,label_skew --rounds 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import pandas as pd
import torch
from flwr.common import Metrics
from flwr.simulation import start_simulation

from metrics import regression_metrics
from partitions import PROTOCOLS, describe_partitions, partition_train
from partition_client import partition_client_fn, write_partition_cache
from train_eval import (
    evaluate_model,
    model_nbytes,
    prepare_xy,
    train_residual_mlp,
)
from utils import (
    TARGET_COL,
    build_forecast_frame,
    dataframe_to_xy,
    fit_feature_scaler,
    naive_seasonal_predict,
    temporal_masks,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SEED = 42


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total = sum(num for num, _ in metrics)
    if total == 0:
        return {}
    keys = metrics[0][1].keys()
    return {
        key: sum(num * m[key] for num, m in metrics if key in m) / total for key in keys
    }


def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_naive(test_df: pd.DataFrame) -> Dict[str, float]:
    pred = naive_seasonal_predict(test_df)
    return regression_metrics(test_df[TARGET_COL].values, pred)


def run_centralized(train_df, val_df, test_df, x_scaler) -> Dict[str, float]:
    X_train, y_train, X_val, y_val, X_test, y_test, lag12 = prepare_xy(
        train_df, test_df, x_scaler, val_df=val_df
    )
    model = train_residual_mlp(X_train, y_train, X_val, y_val)
    return evaluate_model(model, X_test, y_test, lag12)


def run_local_only(parts, val_df, test_df, x_scaler) -> Dict[str, float]:
    """Train one model per partition; size-weighted metrics on the shared test set."""
    lag12 = test_df["gst_lag_12"].values.astype(float)
    X_test, y_test = dataframe_to_xy(test_df, x_scaler)

    weighted = {"rmse": 0.0, "mae": 0.0, "mape": 0.0}
    total_n = 0
    for pid in sorted(parts):
        train_df = parts[pid]
        if len(train_df) < 5:
            continue
        X_train, y_train, X_val, y_val, _, _, _ = prepare_xy(
            train_df, test_df, x_scaler, val_df=val_df
        )
        model = train_residual_mlp(X_train, y_train, X_val, y_val, epochs=300, patience=25)
        m = evaluate_model(model, X_test, y_test, lag12)
        n = len(train_df)
        for k in weighted:
            weighted[k] += m[k] * n
        total_n += n
    if total_n == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "mape": float("nan")}
    return {k: weighted[k] / total_n for k in weighted}


def run_federated(
    parts,
    test_df,
    x_scaler,
    num_rounds: int,
    local_epochs: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Returns (final_test_metrics, aux) where aux has rounds_to_best and bytes_est.
    """
    # Drop empty partitions and reindex 0..K-1 for Flower.
    nonempty = {i: parts[k] for i, k in enumerate(sorted(parts)) if len(parts[k]) > 0}
    num_clients = len(nonempty)
    if num_clients < 2:
        raise RuntimeError("Need at least 2 non-empty clients for FL")

    write_partition_cache(nonempty, test_df, x_scaler, local_epochs)

    nbytes = model_nbytes()
    # Each round: all clients upload + download parameters.
    bytes_per_round = num_clients * 2 * nbytes

    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=min(5, num_clients),
        min_available_clients=num_clients,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
    )

    hist = start_simulation(
        client_fn=partition_client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    rmse_hist = hist.metrics_distributed.get("rmse", [])
    best_round, best_rmse = None, float("inf")
    for round_num, value in rmse_hist:
        if value < best_rmse:
            best_rmse = value
            best_round = round_num

    final = {}
    for key in ("rmse", "mae", "mape"):
        series = hist.metrics_distributed.get(key, [])
        final[key] = float(series[-1][1]) if series else float("nan")

    aux = {
        "fl_rounds": num_rounds,
        "fl_rounds_to_best_rmse": int(best_round) if best_round is not None else -1,
        "fl_bytes_est": int(bytes_per_round * num_rounds),
        "fl_clients": num_clients,
        "model_nbytes": nbytes,
    }
    return final, aux


def run_protocol(
    protocol: str,
    train_df,
    val_df,
    test_df,
    x_scaler,
    num_rounds: int,
    local_epochs: int,
) -> List[Dict]:
    parts = partition_train(train_df, protocol)
    print(f"\n=== Protocol: {protocol} | {describe_partitions(parts)} ===")

    naive = run_naive(test_df)
    central = run_centralized(train_df, val_df, test_df, x_scaler)
    local = run_local_only(parts, val_df, test_df, x_scaler)
    fl_metrics, fl_aux = run_federated(parts, test_df, x_scaler, num_rounds, local_epochs)

    rows = []
    for method, metrics, extra in [
        ("naive_seasonal", naive, {}),
        ("centralized", central, {}),
        ("local_only", local, {}),
        ("federated_fedavg", fl_metrics, fl_aux),
    ]:
        row = {
            "protocol": protocol,
            "method": method,
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "mape": metrics["mape"],
            "seed": SEED,
            **extra,
        }
        rows.append(row)
        print(
            f"  {method:<18} RMSE={metrics['rmse']:.2f}  "
            f"MAE={metrics['mae']:.2f}  MAPE%={metrics['mape']:.2f}"
            + (
                f"  rounds_to_best={extra.get('fl_rounds_to_best_rmse')}  "
                f"bytes≈{extra.get('fl_bytes_est')}"
                if method == "federated_fedavg"
                else ""
            )
        )
    return rows


def print_markdown_table(df: pd.DataFrame):
    cols = [
        "protocol",
        "method",
        "rmse",
        "mae",
        "mape",
        "fl_rounds_to_best_rmse",
        "fl_bytes_est",
    ]
    cols = [c for c in cols if c in df.columns]
    print("\n### Phase 2 comparison table\n")
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    print(header)
    print(sep)
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            val = row[c]
            if c in ("rmse", "mae", "mape") and pd.notna(val):
                cells.append(f"{val:.2f}")
            elif pd.isna(val):
                cells.append("")
            else:
                cells.append(str(val))
        print("| " + " | ".join(cells) + " |")


def main():
    parser = argparse.ArgumentParser(description="Phase 2 FL vs central vs local comparison")
    parser.add_argument(
        "--protocols",
        type=str,
        default="by_state,iid,label_skew,quantity_skew",
        help="Comma-separated protocols",
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=5)
    args = parser.parse_args()

    protocols = [p.strip() for p in args.protocols.split(",") if p.strip()]
    for p in protocols:
        if p not in PROTOCOLS and p != "feature_skew":
            raise SystemExit(f"Unknown protocol {p}. Valid: {PROTOCOLS}")

    set_seed(SEED)
    df = build_forecast_frame()
    train_mask, val_mask, test_mask = temporal_masks(df)
    train_df = df.loc[train_mask].reset_index(drop=True)
    val_df = df.loc[val_mask].reset_index(drop=True)
    test_df = df.loc[test_mask].reset_index(drop=True)
    x_scaler = fit_feature_scaler(train_df)

    print("Phase 2 non-IID comparison")
    print(f"Train={len(train_df)} Val={len(val_df)} Test={len(test_df)} seed={SEED}")
    print(f"FL rounds={args.rounds} local_epochs={args.local_epochs}")

    all_rows = []
    for protocol in protocols:
        set_seed(SEED)
        all_rows.extend(
            run_protocol(
                protocol,
                train_df,
                val_df,
                test_df,
                x_scaler,
                num_rounds=args.rounds,
                local_epochs=args.local_epochs,
            )
        )

    results = pd.DataFrame(all_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "phase2_comparison.csv"
    json_path = RESULTS_DIR / "phase2_comparison.json"
    results.to_csv(csv_path, index=False)
    results.to_json(json_path, orient="records", indent=2)

    print_markdown_table(results)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
