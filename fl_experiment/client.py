import flwr as fl
import torch
import torch.optim as optim
from model import RevenueModel
from utils import load_state_data

class StateClient(fl.client.NumPyClient):
    def __init__(self, state_id):
        self.model = RevenueModel()
        self.X, self.y = load_state_data(state_id)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_fn = torch.nn.MSELoss()

    def get_parameters(self, config):
        return [val.detach().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        for p, new_p in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(new_p)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        for _ in range(5):  # local epochs
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(self.X), self.y)
            loss.backward()
            self.optimizer.step()

        return self.get_parameters(config), len(self.X), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss = self.loss_fn(self.model(self.X), self.y)
        return float(loss), len(self.X), {}

def client_fn(cid):
    return StateClient(state_id=int(cid) + 1)

fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=client_fn
)