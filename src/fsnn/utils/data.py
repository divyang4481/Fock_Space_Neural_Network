"""Dataset loading utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


class DatasetInfo(Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, int, int]):
    """Tuple: (train_ds, test_ds, input_dim, num_classes)."""


def load_mnist(root: Path) -> DatasetInfo:
    transform = transforms.ToTensor()
    train = datasets.MNIST(root=str(root), train=True, download=True, transform=transform)
    test = datasets.MNIST(root=str(root), train=False, download=True, transform=transform)
    return train, test, 28 * 28, 10


def load_adult(root: Path) -> DatasetInfo:
    data = fetch_openml("adult", version=2, as_frame=True, data_home=root)
    df = data.frame
    X = df.drop(columns=["class"])
    y = df["class"].map({"<=50K": 0, ">50K": 1}).astype("int64")
    cat_cols = X.select_dtypes(include="category").columns
    num_cols = X.select_dtypes(exclude="category").columns
    enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    X_cat = enc.fit_transform(X[cat_cols])
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X[num_cols])
    import numpy as np

    X_all = np.hstack([X_num, X_cat]).astype("float32")
    y_all = y.to_numpy().astype("int64")
    X_tensor = torch.from_numpy(X_all)
    y_tensor = torch.from_numpy(y_all)
    dataset = TensorDataset(X_tensor, y_tensor)
    n_train = int(0.8 * len(dataset))
    train, test = torch.utils.data.random_split(dataset, [n_train, len(dataset) - n_train])
    return train, test, X_tensor.shape[1], 2


LOADERS = {
    "mnist": load_mnist,
    "adult": load_adult,
}


def get_dataloaders(
    name: str,
    root: Path,
    batch_size: int = 64,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, int, int]:
    loader = LOADERS[name]
    train_ds, test_ds, input_dim, num_classes = loader(root)
    if name == "mnist":
        def flatten(x):
            return x.view(x.size(0), -1)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        # We'll flatten in training loop
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader, input_dim, num_classes


__all__ = ["get_dataloaders"]
