from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from molfeat.calc.atom import AtomCalculator
from molfeat.calc.bond import EdgeMatCalculator
from molfeat.trans.graph.adj import PYGGraphTransformer
from sklearn.metrics import roc_auc_score
from torch.nn import BCELoss
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data.data import Data
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.models import PNA
from torch_geometric.utils import degree
from tqdm import tqdm

from graph.mol_dataset import build_dataset
from pipeline import Pipeline


class PNAPipeline(Pipeline):
    DEVICE = "cpu"
    NUM_EPOCHS = 120
    LEARNING_RATE = 5e-4
    PNA_AGGREGATORS = ["mean", "min", "max", "std"]
    PNA_SCALERS = ["identity", "amplification", "attenuation"]

    def __init__(self):
        self._transformer = PYGGraphTransformer(atom_featurizer=AtomCalculator(), bond_featurizer=EdgeMatCalculator())
        self._transformer.auto_self_loop()
        self._model = None

    def transform_smiles(self, smiles: np.array) -> np.array:
        return self._transformer(smiles)

    def split_data(self, X, y, train_indices, test_indices):
        X_train = [X[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]
        y_train = [y[i] for i in train_indices]
        y_test = [y[i] for i in test_indices]
        return X_train, X_test, y_train, y_test

    @property
    def num_atom_features(self):
        return self._transformer.atom_dim

    @property
    def num_bond_features(self):
        return self._transformer.bond_dim

    def _get_model(self, X: List[Data] = None):
        if self._model is None:
            self._model = PNA(
                in_channels=self.num_atom_features,
                hidden_channels=64,
                num_layers=2,
                out_channels=1,
                dropout=0.2,
                act="relu",
                edge_dim=self.num_bond_features,
                aggregators=self.PNA_AGGREGATORS,
                scalers=self.PNA_SCALERS,
                deg=self._get_degree(X),
            )
        return self._model

    def _get_degree(self, X: List[Data]):
        max_degree = -1
        for data in X:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(d.max()))
        # Compute the in-degree histogram tensor
        deg = torch.zeros(max_degree + 1, dtype=torch.long)
        for data in X:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
        return deg

    def forward(self, data: Data):
        out = self._model(data.x, data.edge_index, edge_attr=data.edge_attr)
        out = global_add_pool(out, data.batch)
        return F.sigmoid(out)

    def fit(self, X, y):
        train_dt, train_loader = build_dataset(self._transformer, X, y, shuffle=True)
        model = self._get_model(X)

        loss_fn = BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.LEARNING_RATE)
        lr_scheduler = StepLR(optimizer, step_size=10, gamma=1.0)
        with tqdm(range(self.NUM_EPOCHS)) as pbar:
            for epoch in pbar:
                losses = []
                y_hat = []
                y_true = []
                model.train()
                for data in train_loader:
                    data = data.to(self.DEVICE)
                    optimizer.zero_grad()
                    out = self.forward(data)
                    loss = loss_fn(out.squeeze(), data.y)
                    loss.backward()
                    optimizer.step()
                    # save metrics
                    losses.append(loss.item())
                    y_hat.append(out.detach().cpu().squeeze())
                    y_true.append(data.y)

                lr_scheduler.step()

                mean_train_loss = np.mean(losses)
                y_hat = torch.cat(y_hat).numpy()
                y_true = torch.cat(y_true).numpy()
                auc = roc_auc_score(y_true, y_hat)
                pbar.set_description(f"Epoch {epoch} - Train Loss {mean_train_loss:.3f} - Train AUC {auc:.3f}")

    def predict_proba(self, X, y):
        _, test_loader = build_dataset(self._transformer, X, y, shuffle=False)
        model = self._get_model()
        model.eval()
        preds = []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.DEVICE)
                out = self.forward(data)
                preds.append(out.detach().cpu().squeeze())
        return torch.cat(preds).numpy()
