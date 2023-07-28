import flwr as fl
import torch
from model.model import BaseNet
from config import RUN_ID
from torch.utils.data import DataLoader
from utils import plot_tensorboard

from torch.utils.tensorboard import SummaryWriter

tensorboard_writer = SummaryWriter(RUN_ID)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net: BaseNet, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return self.net.get_parameters()

    def fit(self, parameters, config):
        # Read values from config
        server_round = config["server_round"]
        local_epochs = config["local_epochs"]
        lr = config["learning_rate"]

        # Use values provided by the config
        print(
            f"[Client {self.cid}, round {server_round}] fit, config: {config}")
        self.net.set_parameters(parameters)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.net.train_epoch(
            self.trainloader, epochs=local_epochs, optimizer=optimizer)
        return self.net.get_parameters(), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        self.net.set_parameters(parameters)
        loss, accuracy, precision, confusion_matrix = self.net.test(
            self.valloader)
        server_round = config["server_round"]

        plot_tensorboard(tensorboard_writer, loss, accuracy, precision,
                         confusion_matrix, f"client-test/{self.cid}", server_round)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy), "precision": precision, "confusion_matrix": confusion_matrix}


def client_fn(cid, net, trainloaders: list[DataLoader], valloaders: list[DataLoader]) -> FlowerClient:
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    return FlowerClient(cid, net, trainloader, valloader)
