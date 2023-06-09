from dataloader import load_cifars
import flwr as fl
from config import DEVICE, NUM_CLIENTS
from client import client_fn
from flwr.server.strategy import Strategy


def simulate_cifar(strategy: Strategy):
    trainloaders, valloaders, testloader = load_cifars(NUM_CLIENTS)

    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    client_resources = None
    if DEVICE.type == "cuda":
        client_resources = {"num_gpus": 1}

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=lambda cid: client_fn(cid, trainloaders, valloaders),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=3),  # Just three rounds
        strategy=strategy,
        client_resources=client_resources,
    )

