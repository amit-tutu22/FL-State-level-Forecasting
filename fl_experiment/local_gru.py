"""Local baseline: one GRU per state, trained only on that state's windows (#17).

    python -m fl_experiment.local_gru
    python -m fl_experiment.local_gru --source panel --allow-assumed-vintages \
        --panel data/reports/lagged45-2012/panel.csv --features gst_only

Early stopping uses the state's validation windows; the test period is never read.
"""

import argparse
import copy
import csv
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from fl_experiment.metrics import regression_metrics
from fl_experiment.model import RevenueModel
from fl_experiment.server import MetricsRecorder
from fl_experiment.state_data import add_data_arguments, clients_from_arguments

RESULTS = Path(__file__).resolve().parent / "results"


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def train_arrays(x_train, y_train, x_val, y_val, n_features, settings):
    """Train with early stopping on `x_val` and restore the best weights."""
    torch.manual_seed(settings["seed"])
    model = RevenueModel(n_features, settings["hidden"], settings["dropout"])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]
    )
    loss_fn = torch.nn.MSELoss()
    best = {"loss": float("inf"), "epoch": 0, "state": copy.deepcopy(model.state_dict())}
    stale, epoch = 0, 0
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        loss_fn(model(x_train), y_train).backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val), y_val).item())
        if val_loss < best["loss"] - 1e-6:
            best = {"loss": val_loss, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
            stale = 0
        else:
            stale += 1
            if stale >= settings["patience"]:
                break
    model.load_state_dict(best["state"])
    model.eval()
    return model, {"best_epoch": best["epoch"], "epochs_run": epoch, "best_val_loss": best["loss"]}


def train_state(data, settings):
    """Train on the fixed training split, early-stopping on the fixed validation split."""
    return train_arrays(
        torch.from_numpy(data.x["train"]),
        torch.from_numpy(data.y["train"]),
        torch.from_numpy(data.x["val"]),
        torch.from_numpy(data.y["val"]),
        data.n_features,
        settings,
    )


def predict(model, data, split):
    """Return (actual, predicted) in INR crore for one split."""
    with torch.no_grad():
        prediction = model(torch.from_numpy(data.x[split])).numpy()
    return data.decode(data.y[split]), data.decode(prediction)


def train_all(clients, settings):
    """Return (prediction rows, (count, metrics) pairs, per-state record) for all states."""
    rows, results, per_state = [], [], {}
    for index, data in enumerate(clients):
        model, training = train_state(data, dict(settings, seed=settings["seed"] + index))
        for split in ("train", "val"):
            actual, predicted = predict(model, data, split)
            rows += [
                [data.state_id, split, month, f"{a:.4f}", f"{p:.4f}"]
                for month, a, p in zip(data.months[split], actual, predicted)
            ]
        metrics = regression_metrics(*predict(model, data, "val"))
        per_state[data.state_id] = {**metrics, **training, "windows": data.summary()}
        results.append((len(data.months["val"]), metrics))
    return rows, results, per_state


def add_training_arguments(parser):
    """Register the shared GRU training options."""
    parser.add_argument("--epochs", type=int, default=500, help="maximum epochs")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def training_settings(args):
    """Return the training options as a dictionary."""
    return {
        key: getattr(args, key)
        for key in ["epochs", "patience", "lr", "weight_decay", "hidden", "dropout", "seed"]
    }


def compute_info(started):
    """Wall time and the compute environment for a run record."""
    return {
        "wall_seconds": round(time.perf_counter() - started, 2),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "threads": 1,
    }


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    add_training_arguments(parser)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main():
    """Train every state's GRU and export predictions and metrics."""
    args = parse_args()
    torch.set_num_threads(1)
    clients, _, manifest = clients_from_arguments(args)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir or RESULTS / f"local-gru-{args.source}-{args.features}-{stamp}"
    output.mkdir(parents=True, exist_ok=True)
    settings = training_settings(args)

    started = time.perf_counter()
    rows, results, per_state = train_all(clients, settings)
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["state_id", "split", "target_month", "actual_crore", "predicted_crore"])
        writer.writerows(rows)
    record = {
        "created_utc": stamp,
        "model": "local GRU, one per state",
        "settings": settings,
        "seed_rule": "seed + client index",
        "eval_split": "val",
        "data": manifest,
        "validation_aggregate": MetricsRecorder()(results),
        "validation_by_state": per_state,
        "compute": compute_info(started),
    }
    (output / "metrics.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"{len(clients)} states; excluded: {manifest['excluded_states'] or 'none'}")
    print(f"Validation (INR crore): {json.dumps(record['validation_aggregate'])}")
    print(f"Outputs: {output}")


if __name__ == "__main__":
    main()
