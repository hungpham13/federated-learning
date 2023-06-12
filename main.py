from dataloader import load_cifars
import flwr as fl
from config import DEVICE, NUM_CLIENTS, CLASSES
from client import client_fn
from flwr.server.strategy import Strategy
from flwr.common import Metrics
from typing import List, Tuple, Optional, Dict, Type
from model import BaseNet
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    precisions = {}
    for label in metrics[0][1]["precision"]:
        nominator = sum([num_examples * m["precision"][label] 
                  for num_examples, m in metrics if m["precision"][label] is not None])
        denominator = sum([num_examples for num_examples, m in metrics if m["precision"][label] is not None])
        precisions[label] = nominator / denominator if denominator != 0 else None

    # Aggregate and return custom metric (weighted average)
    return {
        "accuracy": sum(accuracies) / sum(examples),
        "precision": precisions,
    }


def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
        net: BaseNet,
        testloader: DataLoader,
        tensorboard_writer: SummaryWriter,
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    net.set_parameters(parameters)  # Update model with the latest parameters
    loss, accuracy, precision = net.test(testloader)
    print(
        f"Server-side evaluation loss {loss} / accuracy {accuracy} / precision {precision}")
    tensorboard_writer.add_scalar("Loss/server-test", loss, server_round)
    tensorboard_writer.add_scalar("Accuracy/server-test", accuracy, server_round)
    for label in precision:
        if precision[label] is not None:
            tensorboard_writer.add_scalar(f"Precision {label}/server-test", precision[label], server_round)

    return loss, {"accuracy": accuracy, "precision": precision}


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterward.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
        "local_epochs": 1 if server_round < 2 else 2,  #
    }
    return config


def simulate(StrategyCls: Type[Strategy], strategyArgs, net, loaders, num_rounds=3):
    trainloaders, valloaders, testloader = loaders
    tensorboard_writer = SummaryWriter()

    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    client_resources = None
    if DEVICE.type == "cuda":
        client_resources = {"num_gpus": 1}

    strategy = StrategyCls(
        # <-- pass the metric aggregation function
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=fl.common.ndarrays_to_parameters(
            net.get_parameters()),
        evaluate_fn=lambda x, y, z: evaluate(x, y, z, net, testloader, tensorboard_writer),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=fit_config,
        **strategyArgs,
    )

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=lambda cid: client_fn(cid, net, trainloaders, valloaders),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )
