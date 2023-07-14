from typing import List

import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from config import DEVICE, CLASSES, CLASS_WEIGHTS
from .focal_loss import focal_loss


class BaseNet(nn.Module):
    def __init__(self, focus_labels=[0]):
        super(BaseNet, self).__init__()
        self.focus_labels = focus_labels

    # def scheduler_step(self):
    #     self.scheduler.step()

    def get_parameters(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: tensor(v) for k, v in params_dict})
        self.load_state_dict(state_dict, strict=True)

    def train_epoch(self, trainloader, epochs: int, optimizer, scheduler=None):
        """Train the network on the training set."""
        criterion = torch.nn.CrossEntropyLoss()
        # criterion = focal_loss(alpha=CLASS_WEIGHTS, gamma=2)

        # optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.8)
        self.train()
        self.to(DEVICE)
        for epoch in range(epochs):
            correct, total, epoch_loss = 0, 0, 0.0
            confusion_matrix = np.zeros((len(CLASSES), len(CLASSES)))
            for images, labels in trainloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(self(images), labels)
                loss.backward()
                optimizer.step()
                # Metrics
                epoch_loss += loss
                total += labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                for p in range(len(CLASSES)):
                    for t in range(len(CLASSES)):
                        confusion_matrix[p][t] += torch.logical_and(
                            predicted == p, labels == t).sum().item()

            # Metrics
            epoch_loss /= len(trainloader.dataset)
            epoch_acc = correct / total
            confusion_matrix = confusion_matrix / \
                np.sum(confusion_matrix, axis=1)[:, None]
            epoch_precision = {CLASSES[l]: confusion_matrix[l][l]
                               for l in self.focus_labels}
            print(
                f"Epoch {epoch + 1}: train loss {epoch_loss}, accuracy {epoch_acc}")
            print("\tprecision:", epoch_precision)
            print("\tconfusion matrix:", confusion_matrix)

    def test(self, testloader):
        """Evaluate the network on the entire test set."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        confusion_matrix = np.zeros((len(CLASSES), len(CLASSES)))
        self.eval()
        self.to(DEVICE)
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                for p in range(len(CLASSES)):
                    for t in range(len(CLASSES)):
                        confusion_matrix[p][t] += torch.logical_and(
                            predicted == p, labels == t).sum().item()

        loss /= len(testloader.dataset)
        accuracy = correct / total
        confusion_matrix = confusion_matrix / \
            np.sum(confusion_matrix, axis=1)[:, None]
        precision = {CLASSES[l]: confusion_matrix[l][l]
                     for l in self.focus_labels}
        return loss, accuracy, precision, confusion_matrix


class Net(BaseNet):
    def __init__(self, num_classes=10, focus_labels=[0]) -> None:
        super(Net, self).__init__(focus_labels)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)  # 6 62 62
        self.conv2 = nn.Conv2d(6, 16, 5)  # 16 29 29
        self.fc1 = nn.Linear(16 * 29 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> tensor:
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        x = x.view(-1, 16 * 29 * 29)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
