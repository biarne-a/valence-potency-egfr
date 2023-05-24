from typing import Tuple

import numpy as np
import torch
from molfeat.trans.graph.adj import PYGGraphTransformer
from torch.utils.data import DataLoader, Dataset


class MolDataset(Dataset):
    def __init__(self, transformer, X: np.array, y: np.array):
        super().__init__()
        self._transformer = transformer
        self._X = X
        self._y = torch.tensor(y).unsqueeze(-1).float()

    def collate_fn(self, **kwargs):
        return self._transformer.get_collate_fn(**kwargs)

    def __len__(self):
        return len(self._X)

    def __getitem__(self, index):
        return self._X[index], self._y[index]


def build_dataset(
    transformer: PYGGraphTransformer, X: np.array, y: np.array, shuffle: bool = False, batch_size=32
) -> Tuple[MolDataset, DataLoader]:
    dt = MolDataset(transformer, X, y)
    loader = DataLoader(dt, batch_size=batch_size, shuffle=shuffle, collate_fn=dt.collate_fn(return_pair=False))
    return dt, loader
