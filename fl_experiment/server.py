"""Optional standalone Flower server for multi-process client runs.

Prefer run_fl_simulation.py for the Phase 0 single-process demo.
"""

import flwr as fl

strategy = fl.server.strategy.FedAvg(
    min_fit_clients=5,
    min_available_clients=5,
    fraction_fit=1.0,
    fraction_evaluate=1.0,
)

if __name__ == "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=10),
    )
