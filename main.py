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
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
import torch
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
            # nominator = sum([num_examples * m["confusion_matrix"][p][t]
            #                  for num_examples, m in metrics if not np.isnan(m["confusion_matrix"][p][t])])
            # denominator = sum([num_examples for num_examples, m in metrics if not np.isnan(
            #     m["confusion_matrix"][p][t])])
            # confusion_matrix[p][t] = nominator / \
            #     denominator if denominator != 0 else np.nan
            confusion_matrix[p][t] = sum(
                [m["confusion_matrix"][p][t] for num_examples, m in metrics if not np.isnan(m["confusion_matrix"][p][t])])

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


def fit_config(server_round: int, learning_rate: float):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterward.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
        "local_epochs": 1 if server_round < 2 else 2,  #
        "learning_rate": learning_rate,
    }
    return config


def simulate(StrategyCls: Type[Strategy], strategyArgs, net, loaders, num_rounds=3, learning_rate=1e-6, scheduler=None):
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
        on_fit_config_fn=lambda x: fit_config(x, learning_rate),
        on_evaluate_config_fn=lambda x: fit_config(x, learning_rate),
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


def centralize_training(net: BaseNet, loaders, epoch_num, optimizer, scheduler=None):
    trainloader, valloader, testloader = loaders

    tensor_writer = SummaryWriter(RUN_ID)

    for epoch in range(epoch_num):
        net.train_epoch(trainloader, 1, optimizer=optimizer)
        loss, accuracy, precision, confusion_matrix = net.test(valloader)
        plot_tensorboard(tensor_writer, loss, accuracy, precision,
                         confusion_matrix, "centralize-train-validation", epoch)
        try:
            learning_rate = scheduler._last_lr[0]
        except:
            learning_rate = optimizer.param_groups[0]['lr']

        tensor_writer.add_scalar(f"Learning Rate", learning_rate, epoch)

        print(
            f"Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}, precision {precision}, learning rate: {learning_rate}")

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(loss)
        elif scheduler is not None:
            scheduler.step()

    loss, accuracy, precision, confusion_matrix = net.test(testloader)
    plot_tensorboard(tensor_writer, loss, accuracy, precision,
                     confusion_matrix, "centralize-test", 0)
    print(
        f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}\n\tprecision {precision}\n\tconfusion matrix {confusion_matrix}")
