from dataloader import load_cifars
import flwr as fl
from config import DEVICE, NUM_CLIENTS, RUN_ID, CLASSES
from client import client_fn
from flwr.server.strategy import Strategy
from flwr.common import Metrics
from typing import List, Tuple, Optional, Dict, Type
from model.model import BaseNet
from torch.utils.data import DataLoader
from utils import plot_tensorboard
import numpy as np
from dataloader import load_fitzpatrick
from torch.utils.tensorboard import SummaryWriter


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    precisions = {}
    confusion_matrix = np.zeros((len(CLASSES), len(CLASSES)))
    for label in metrics[0][1]["precision"]:
        nominator = sum([num_examples * m["precision"][label]
                         for num_examples, m in metrics if m["precision"][label] is not None])
        denominator = sum([num_examples for num_examples,
                          m in metrics if m["precision"][label] is not None])
        precisions[label] = nominator / \
            denominator if denominator != 0 else None

    for p in range(len(CLASSES)):
        for t in range(len(CLASSES)):
            nominator = sum([num_examples * m["confusion_matrix"][p][t]
                             for num_examples, m in metrics if not np.isnan(m["confusion_matrix"][p][t])])
            denominator = sum([num_examples for num_examples, m in metrics if not np.isnan(
                m["confusion_matrix"][p][t])])
            confusion_matrix[p][t] = nominator / \
                denominator if denominator != 0 else np.nan

    # Aggregate and return custom metric (weighted average)
    return {
        "accuracy": sum(accuracies) / sum(examples),
        "precision": precisions,
        "confusion_matrix": confusion_matrix,
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
    loss, accuracy, precision, confusion_matrix = net.test(testloader)
    print(
        f"Server-side evaluation loss {loss} / accuracy {accuracy} / precision {precision}")
    plot_tensorboard(tensorboard_writer, loss, accuracy, precision,
                     confusion_matrix, "server-test", server_round)

    return loss, {"accuracy": accuracy, "precision": precision, "confusion_matrix": confusion_matrix}


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
    tensorboard_writer = SummaryWriter(RUN_ID)

    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    client_resources = None
    if DEVICE.type == "cuda":
        client_resources = {"num_gpus": 1}

    strategy = StrategyCls(
        # <-- pass the metric aggregation function
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=fl.common.ndarrays_to_parameters(
            net.get_parameters()),
        evaluate_fn=lambda x, y, z: evaluate(
            x, y, z, net, testloader, tensorboard_writer),
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


def centralize_training(net, data_loader=load_fitzpatrick, batch_size=32, skin_stratify_sampling=True, epoch_num=30):
    trainloaders, valloaders, testloader = data_loader(
        1, skin_stratify_sampling=skin_stratify_sampling, batch_size=batch_size)

    trainloader = trainloaders[0]
    valloader = valloaders[0]
    tensor_writer = SummaryWriter(RUN_ID)

    for epoch in range(epoch_num):
        net.train_epoch(trainloader, 1)
        loss, accuracy, precision, confusion_matrix = net.test(valloader)
        plot_tensorboard(tensor_writer, loss, accuracy, precision,
                         confusion_matrix, "centralize-train-validation", epoch)
        print(
            f"Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}, precision {precision}")

    loss, accuracy, precision, confusion_matrix = net.test(testloader)
    plot_tensorboard(tensor_writer, loss, accuracy, precision,
                     confusion_matrix, "centralize-test", 0)
    print(
        f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}\n\tprecision {precision}\n\tconfusion matrix {confusion_matrix}")
