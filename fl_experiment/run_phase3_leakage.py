"""Phase 3: welfare leakage detection — centralized + federated baselines.

Usage:
    python run_phase3_leakage.py
    python run_phase3_leakage.py --rounds 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import pandas as pd
import torch
from flwr.common import Metrics
from flwr.simulation import start_simulation

from leakage_client import leakage_client_fn, write_leakage_cache
from leakage_train import (
    choose_threshold,
    eval_scores,
    predict_with_threshold,
    reconstruction_scores,
    score_isolation_forest,
    train_autoencoder,
    train_isolation_forest,
)
from leakage_utils import (
    SEED,
    fit_scaler,
    labels,
    load_welfare_dataframe,
    normal_only,
    state_partitions,
    temporal_split,
    to_tensor,
    transform_X,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    total = sum(num for num, _ in metrics)
    if total == 0:
        return {}
    out = {}
    for key in metrics[0][1]:
        vals = [(num, m[key]) for num, m in metrics if key in m and m[key] == m[key]]
        if not vals:
            continue
        s = sum(num for num, _ in vals)
        out[key] = sum(num * v for num, v in vals) / s
    return out


def run_central_isolation_forest(train_df, test_df, scaler, contamination: float) -> Dict[str, float]:
    X_normal = transform_X(normal_only(train_df), scaler)
    X_test = transform_X(test_df, scaler)
    model = train_isolation_forest(X_normal, contamination=contamination, seed=SEED)
    pred, scores = score_isolation_forest(model, X_test)
    return eval_scores(labels(test_df), pred, scores)


def run_central_autoencoder(train_df, val_df, test_df, scaler) -> Tuple[Dict[str, float], float]:
    X_train = to_tensor(transform_X(normal_only(train_df), scaler))
    X_val = to_tensor(transform_X(val_df, scaler))
    X_test = to_tensor(transform_X(test_df, scaler))
    model = train_autoencoder(X_train, X_val)
    val_scores = reconstruction_scores(model, X_val)
    threshold = choose_threshold(val_scores, labels(val_df))
    test_scores = reconstruction_scores(model, X_test)
    pred = predict_with_threshold(test_scores, threshold)
    return eval_scores(labels(test_df), pred, test_scores), threshold


def run_local_isolation_forest(train_df, test_df, scaler, contamination: float) -> Dict[str, float]:
    parts = state_partitions(train_df)
    X_test = transform_X(test_df, scaler)
    y_test = labels(test_df)
    # Majority vote across state models (each sees only its state normals).
    votes = np.zeros(len(test_df), dtype=float)
    score_sum = np.zeros(len(test_df), dtype=float)
    used = 0
    for sid, part in parts.items():
        normal = normal_only(part)
        if len(normal) < 20:
            continue
        model = train_isolation_forest(
            transform_X(normal, scaler), contamination=contamination, seed=SEED + sid
        )
        pred, scores = score_isolation_forest(model, X_test)
        votes += pred
        score_sum += scores
        used += 1
    if used == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": float("nan"),
                "anomaly_rate_pred": 0.0, "anomaly_rate_true": float(y_test.mean())}
    pred = (votes >= max(1, used / 2)).astype(int)
    scores = score_sum / used
    return eval_scores(y_test, pred, scores)


def run_federated_autoencoder(
    train_df,
    val_df,
    test_df,
    scaler,
    threshold: float,
    num_rounds: int,
    local_epochs: int,
) -> Dict[str, float]:
    parts = state_partitions(train_df)
    # Reindex Flower partitions 0..n-1 but keep state_id filenames via write cache.
    write_leakage_cache(parts, val_df, test_df, scaler, threshold, local_epochs)
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
        client_fn=leakage_client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    out = {}
    for key in ("precision", "recall", "f1", "roc_auc"):
        series = hist.metrics_distributed.get(key, [])
        out[key] = float(series[-1][1]) if series else float("nan")
    return out


def print_table(rows: List[Dict]):
    cols = ["method", "precision", "recall", "f1", "roc_auc"]
    print("\n### Phase 3 leakage detection\n")
    print("| " + " | ".join(cols) + " |")
    print("| " + " | ".join("---" for _ in cols) + " |")
    for row in rows:
        cells = [str(row["method"])]
        for c in cols[1:]:
            v = row.get(c, float("nan"))
            cells.append(f"{v:.3f}" if v == v else "")
        print("| " + " | ".join(cells) + " |")


def main():
    parser = argparse.ArgumentParser(description="Phase 3 welfare leakage detection")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--contamination", type=float, default=0.08)
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    df = load_welfare_dataframe()
    train_df, val_df, test_df = temporal_split(df)
    scaler = fit_scaler(train_df)

    print("Phase 3 welfare leakage detection")
    print(
        f"Rows train={len(train_df)} val={len(val_df)} test={len(test_df)} | "
        f"test anomaly rate={labels(test_df).mean():.3f}"
    )

    if_metrics = run_central_isolation_forest(
        train_df, test_df, scaler, contamination=args.contamination
    )
    print(
        f"  central_isolation_forest  P={if_metrics['precision']:.3f} "
        f"R={if_metrics['recall']:.3f} F1={if_metrics['f1']:.3f} "
        f"AUC={if_metrics['roc_auc']:.3f}"
    )

    ae_metrics, threshold = run_central_autoencoder(train_df, val_df, test_df, scaler)
    print(
        f"  central_autoencoder       P={ae_metrics['precision']:.3f} "
        f"R={ae_metrics['recall']:.3f} F1={ae_metrics['f1']:.3f} "
        f"AUC={ae_metrics['roc_auc']:.3f}  thr={threshold:.4f}"
    )

    local_if = run_local_isolation_forest(
        train_df, test_df, scaler, contamination=args.contamination
    )
    print(
        f"  local_isolation_forest    P={local_if['precision']:.3f} "
        f"R={local_if['recall']:.3f} F1={local_if['f1']:.3f} "
        f"AUC={local_if['roc_auc']:.3f}"
    )

    fl_metrics = run_federated_autoencoder(
        train_df,
        val_df,
        test_df,
        scaler,
        threshold=threshold,
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
    )
    print(
        f"  federated_autoencoder     P={fl_metrics['precision']:.3f} "
        f"R={fl_metrics['recall']:.3f} F1={fl_metrics['f1']:.3f} "
        f"AUC={fl_metrics.get('roc_auc', float('nan')):.3f}"
    )

    rows = [
        {"method": "central_isolation_forest", **if_metrics},
        {"method": "central_autoencoder", **ae_metrics},
        {"method": "local_isolation_forest", **local_if},
        {"method": "federated_autoencoder", **fl_metrics},
    ]
    print_table(rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "phase3_leakage.csv"
    out.to_csv(csv_path, index=False)
    out.to_json(RESULTS_DIR / "phase3_leakage.json", orient="records", indent=2)
    print(f"\nWrote {csv_path}")

    # Soft gate: at least one method should get non-trivial F1 on labeled anomalies.
    best_f1 = max(r["f1"] for r in rows)
    if best_f1 < 0.3:
        raise SystemExit(f"Phase 3 gate failed: best F1={best_f1:.3f} < 0.3")
    print(f"Phase 3 gate passed (best F1={best_f1:.3f})")


if __name__ == "__main__":
    main()
