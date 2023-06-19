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
from config import CLASSES, LABEL_KEY


def download_file(url, dst_path):
    headers = {'User-Agent': 'XY'}
    request = urllib.request.Request(url, headers=headers)

    if (Path(dst_path).exists()):
        return

    with urllib.request.urlopen(request, timeout=120) as web_file:
        data = web_file.read()
        with open(dst_path, mode='wb') as local_file:
            local_file.write(data)


class Fitzpatrick17k(Dataset):
    def __init__(self, csv_file, image_dir, download=False, transform=None):
        df = pd.read_csv(csv_file)
        df.dropna(subset=['url'], inplace=True)
        self.table = df.reset_index(drop=True)

        if download:
            self.error_url = {}
            self.download()
        else:
            self.error_url = None
            downloaded = [os.path.splitext(x)[0] for x in os.listdir(
                image_dir) if os.path.splitext(x)[1] == '.jpg']
            self.table = self.table.loc[self.table['md5hash'].isin(
                downloaded)].reset_index(drop=True)
        self.transform = transform
        self.image_dir = Path(image_dir)

    def __getitem__(self, idx):
        # img_path = self.image_dir / self.table.loc[idx, 'md5hash']
        # if not img_path.exists():
        img_path = self.image_dir / (self.table.loc[idx, 'md5hash'] + '.jpg')

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = CLASSES.index(self.table.loc[idx, LABEL_KEY])
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


def load_fitzpatrick(num_clients: int, skin_stratify_sampling=True, image_dir='./data/images/', skin_seperate=False, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([64, 64]),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = Fitzpatrick17k(
        csv_file='./data/fitzpatrick17k.csv',
        image_dir=image_dir,
        transform=transform,
    )
    df = dataset.table
    if skin_stratify_sampling:
        df_train, df_test = train_test_split(
            df, test_size=0.1, stratify=df[[LABEL_KEY, "fitzpatrick"]], random_state=42)
    else:
        df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)

    if skin_seperate:
        df_train.sort_values('fitzpatrick', inplace=True)
    
    train_set = Subset(dataset, df_train.index)
    test_set = Subset(dataset, df_test.index)

    print('train set loaded, length: ', len(train_set))
    print('test set loaded, length: ', len(test_set))

    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(train_set) // num_clients
    datasets = []
    for i in range(num_clients):
        datasets.append(Subset(train_set, range(
            i * partition_size, (i + 1) * partition_size) if i != num_clients - 1 else range(i * partition_size, len(train_set))))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for i,ds in enumerate(datasets):
        if skin_stratify_sampling:
            table = ds.dataset.dataset.table.loc[ds.indices]
            train_df, val_df = train_test_split(table, stratify=table[[LABEL_KEY, "fitzpatrick"]], test_size=0.1)
            ds_train = Subset(ds, train_df.index)
            ds_val = Subset(ds, val_df.index)
        else:
            len_val = len(ds) // 10  # 10 % validation set
            len_train = len(ds) - len_val
            lengths = [len_train, len_val]
            ds_train, ds_val = random_split(
                ds, lengths, torch.Generator().manual_seed(42))
        print(f'train set {i} loaded, length: ', len(ds_train))
        print('validation set loaded, length: ', len(ds_val))
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
    print('train set loaded, length: ', len(trainset))
    testset = CIFAR10("./dataset", train=False,
                      download=True, transform=transform)
    print('test set loaded, length: ', len(testset))

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
