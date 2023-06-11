import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import torch
import pandas as pd
import urllib.request
import os
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from multiprocessing import Pool


def download_file(url, dst_path):
    headers = {'User-Agent': 'XY'}
    request = urllib.request.Request(url, headers=headers)

    if (Path(dst_path).exists()):
        return

    with urllib.request.urlopen(request, timeout=120) as web_file:
        data = web_file.read()
        with open(dst_path, mode='wb') as local_file:
            local_file.write(data)


CLASSES = ('benign', 'non-neoplastic', 'malignant')


class Fitzpatrick17k(Dataset):
    def __init__(self, csv_file, image_dir, train=True, download=False, transform=None, sort_by_skin_color=False):
        df = pd.read_csv(csv_file)
        df.dropna(subset=['url'], inplace=True)
        df_train, df_test = train_test_split(
            df, test_size=0.1, stratify=df[["fitzpatrick", "three_partition_label"]])

        if train:
            if sort_by_skin_color:
                df_train.sort_values('fitzpatrick', inplace=True)
            self.table = df_train.reset_index(drop=True)
        else:
            self.table = df_test.reset_index(drop=True)

        if download:
            self.error_url = {}
            self.download()
        else:
            self.error_url = None
            downloaded = os.listdir(image_dir)
            self.table = self.table.loc[self.table['md5hash'].isin(
                downloaded)].reset_index(drop=True)
        self.transform = transform
        self.image_dir = Path(image_dir)

    def __getitem__(self, idx):
        img_path = self.image_dir / self.table.loc[idx, 'md5hash']
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = CLASSES.index(self.table.loc[idx, 'three_partition_label'])
        return img, label

    def run_download(self, i):
        url = self.table.loc[i, 'url']
        dst_path = self.image_dir / self.table.loc[i, 'md5hash']
        try:
            download_file(url, dst_path)
        except:
            self.error_url[i] = url

    def download(self):
        if self.image_dir.exists() and len([name for name in os.listdir(self.image_dir)]) == len(self.table):
            print("Files already downloaded")
        else:
            os.makedirs(self.image_dir, exist_ok=True)
            with Pool(4) as p:
                p.map(self.run_download, self.table.index)
            print(self.error_url)

    def __len__(self):
        return len(self.table)


def load_fitzpatrick(num_clients: int, image_dir: str, skin_seperate=False, batch_size=32):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_set = Fitzpatrick17k(
        csv_file='./data/fitzpatrick17k.csv',
        image_dir=image_dir,
        train=True,
        transform=transform,
        sort_by_skin_color=skin_seperate,
    )
    test_set = Fitzpatrick17k(
        csv_file='./data/fitzpatrick17k.csv',
        image_dir=image_dir,
        train=False,
        transform=transform,
    )

    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(train_set) // num_clients
    datasets = []
    for i in range(num_clients):
        datasets.append(Subset(train_set, range(
            i * partition_size, (i + 1) * partition_size) if i != num_clients - 1 else range(i * partition_size, len(train_set))))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(
            ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(
            ds_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))
    testloader = DataLoader(test_set, batch_size=batch_size)
    return trainloaders, valloaders, testloader


def load_cifars(num_clients: int):
    # Download and transform CIFAR-10 (train and test)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10("./dataset", train=True,
                       download=True, transform=transform)
    testset = CIFAR10("./dataset", train=False,
                      download=True, transform=transform)

    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(trainset) // num_clients
    lengths = [partition_size] * num_clients
    datasets = random_split(
        trainset, lengths, torch.Generator().manual_seed(42)
    )

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(
            ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size=32, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=32))
    testloader = DataLoader(testset, batch_size=32)
    return trainloaders, valloaders, testloader
