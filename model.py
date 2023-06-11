from typing import List

import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from config import DEVICE, CLASSES


class BaseNet(nn.Module):
    def __init__(self, focus_label=0):
        super(BaseNet, self).__init__()
        self.focus_label = focus_label

    def get_parameters(self) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        params_dict = zip(self.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: tensor(v) for k, v in params_dict})
        self.load_state_dict(state_dict, strict=True)

    def train_epoch(self, trainloader, epochs: int):
        """Train the network on the training set."""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters())
        self.train()
        for epoch in range(epochs):
            correct, total, epoch_loss = 0, 0, 0.0
            true_pos, label_predicted = 0, 0
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
                true_pos += torch.logical_and(predicted ==
                                              labels, labels == self.focus_label).sum().item()
                label_predicted += (predicted == self.focus_label).sum().item()
            epoch_loss /= len(trainloader.dataset)
            epoch_acc = correct / total
            epoch_precision = true_pos / label_predicted if label_predicted != 0 else None
            print(
                f"Epoch {epoch + 1}: train loss {epoch_loss}, accuracy {epoch_acc}, precision of {CLASSES[self.focus_label]} {epoch_precision}")

    def test(self, testloader):
        """Evaluate the network on the entire test set."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        true_pos, label_predicted = 0, 0
        self.eval()
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = self(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                true_pos += torch.logical_and(predicted ==
                                              labels, labels == self.focus_label).sum().item()
                label_predicted += (predicted == self.focus_label).sum().item()

        loss /= len(testloader.dataset)
        accuracy = correct / total
        precision = true_pos / label_predicted if label_predicted != 0 else None
        return loss, accuracy, precision


class Net(BaseNet):
    def __init__(self, num_classes=10, focus_label=0) -> None:
        super(Net, self).__init__(focus_label)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)  # 6 14 14
        self.conv2 = nn.Conv2d(6, 16, 5)  # 16 5 5
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> tensor:
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape)
        x = x.view(-1, 16 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class VGG16(BaseNet):
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256,
           'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

    def __init__(self, num_classes=10, focus_label=0):
        super(VGG16, self).__init__(focus_label)
        self.features = self._make_layers(self.cfg)
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        out = self.features(x)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
