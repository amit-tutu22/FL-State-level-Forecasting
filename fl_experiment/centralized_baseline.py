"""Pooled (non-federated) reference on the same state clients and splits.

    python -m fl_experiment.centralized_baseline --epochs 200
"""

import argparse
import json

import numpy as np
import torch

from fl_experiment.metrics import regression_metrics
from fl_experiment.model import RevenueModel
from fl_experiment.state_data import add_data_arguments, clients_from_arguments


def main():
    """Train one model on all states' training rows and report per-state validation error."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    clients, _, _ = clients_from_arguments(args)
    x = torch.from_numpy(np.concatenate([c.x["train"] for c in clients]))
    y = torch.from_numpy(np.concatenate([c.y["train"] for c in clients]))
    model = RevenueModel(clients[0].n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    for _ in range(args.epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

    model.eval()
    per_state = {}
    with torch.no_grad():
        for client in clients:
            prediction = model(torch.from_numpy(client.x["val"])).numpy()
            per_state[client.state_id] = regression_metrics(
                client.decode(client.y["val"]), client.decode(prediction)
            )
    print(json.dumps({"validation_by_state": per_state}, indent=2))


if __name__ == "__main__":
    main()
