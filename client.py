import flwr as fl
from model import Net
from config import DEVICE
from torch.utils.data import DataLoader


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net: Net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return self.net.get_parameters()

    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit, config: {config}")
        self.net.set_parameters(parameters)
        self.net.train_epoch(self.trainloader, epochs=1)
        return self.net.get_parameters(), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        self.net.set_parameters(parameters)
        loss, accuracy = self.net.test(self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def client_fn(cid, trainloaders: list[DataLoader], valloaders: list[DataLoader]) -> FlowerClient:
    net = Net().to(DEVICE)
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    return FlowerClient(cid, net, trainloader, valloader)
