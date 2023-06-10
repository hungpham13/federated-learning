import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import pandas as pd
import urllib.request
import os
from pathlib import Path
from tqdm import tqdm


def download_file(img_url: str, path: Path):
    headers = {'User-Agent': 'XY'}
    request = urllib.request.Request(img_url, headers=headers)

    with urllib.request.urlopen(request) as web_file:
        data = web_file.read()
        with open(path, mode='wb') as local_file:
            local_file.write(data)


class Fitzpatrick17k(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        df = pd.read_csv(csv_file)
        df.dropna(subset=['url'], inplace=True)
        self.table = df
        self.transform = transform
        self.image_dir = Path(image_dir)

    def __getitem__(self, idx):
        pass

    def download(self):
        folder_dir = self.image_dir

        if folder_dir.exists():
            print("Files already downloaded")


        os.mkdir(folder_dir)
        error_url = {}

        for i in tqdm(self.table.index):
            url = self.table.loc[i, 'url']
            dst_path = folder_dir / self.table.loc[i, 'md5hash']
            try:
                download_file(url, dst_path)
            except:
                error_url[i] = url

        error_url

    def __len__(self):
        return len(self.table)


def load_cifars(num_clients: int):
    # Download and transform CIFAR-10 (train and test)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
    testset = CIFAR10("./dataset", train=False, download=True, transform=transform)

    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(trainset) // num_clients
    lengths = [partition_size] * num_clients
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size=32, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=32))
    testloader = DataLoader(testset, batch_size=32)
    return trainloaders, valloaders, testloader


def load_fitzpatrick():
    pass
