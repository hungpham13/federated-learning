import flwr as fl
from model import BaseNet
from config import DEVICE
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

tensorboard_writer = SummaryWriter()

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

        # Use values provided by the config
        print(
            f"[Client {self.cid}, round {server_round}] fit, config: {config}")
        self.net.set_parameters(parameters)
        self.net.train_epoch(self.trainloader, epochs=local_epochs)
        return self.net.get_parameters(), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        self.net.set_parameters(parameters)
        loss, accuracy, precision = self.net.test(self.valloader)
        server_round = config["server_round"]
        tensorboard_writer.add_scalar(f"Loss/client-test/{self.cid}", loss, server_round)
        tensorboard_writer.add_scalar(f"Accuracy/client-test/{self.cid}", accuracy, server_round)
        for label in precision:
            if precision[label] is not None:
                tensorboard_writer.add_scalar(f"Precision {label}/client-test/{self.cid}", precision[label], server_round)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy), "precision": precision}


def client_fn(cid, net, trainloaders: list[DataLoader], valloaders: list[DataLoader]) -> FlowerClient:
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    return FlowerClient(cid, net, trainloader, valloader)
