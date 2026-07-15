"""Flower NumPyClient for per-state GST revenue regression."""

from __future__ import annotations

from typing import List

import flwr as fl
import numpy as np
import torch
import torch.optim as optim
from flwr.common import Context

from model import RevenueModel
from utils import load_state_data


class StateClient(fl.client.NumPyClient):
    def __init__(self, state_id: int, local_epochs: int = 5):
        self.state_id = state_id
        self.local_epochs = local_epochs
        self.model = RevenueModel()
        self.X, self.y = load_state_data(state_id)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_fn = torch.nn.MSELoss()

    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters: List[np.ndarray]):
        params_dict = zip(self.model.parameters(), parameters)
        for p, new_p in params_dict:
            p.data = torch.as_tensor(new_p, dtype=p.dtype, device=p.device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        loss = None
        for _ in range(self.local_epochs):
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(self.X), self.y)
            loss.backward()
            self.optimizer.step()
        return self.get_parameters(config), len(self.X), {"loss": float(loss.item())}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        with torch.no_grad():
            loss = self.loss_fn(self.model(self.X), self.y)
        return float(loss.item()), len(self.X), {"mse": float(loss.item())}


def client_fn(context: Context) -> fl.client.Client:
    """Map simulation partition -> state_id (1..10)."""
    partition_id = int(context.node_config["partition-id"])
    return StateClient(state_id=partition_id + 1).to_client()
