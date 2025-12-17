import flwr as fl

strategy = fl.server.strategy.FedAvg(
    min_fit_clients=5,
    min_available_clients=5,
    fraction_fit=1.0,
    fraction_evaluate=1.0
)

fl.server.start_server(
    server_address="localhost:8080",
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=10)
)