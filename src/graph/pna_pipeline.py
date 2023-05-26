from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from molfeat.calc.atom import AtomCalculator
from molfeat.calc.bond import EdgeMatCalculator
from molfeat.trans.graph.adj import PYGGraphTransformer
from sklearn.metrics import roc_auc_score
from torch.nn import BCELoss
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch_geometric.data.data import Data
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.models import PNA
from torch_geometric.utils import degree
from tqdm import tqdm

from graph.augmenter import augmenter_factory
from graph.mol_dataset import build_dataset
from pipeline import Pipeline


class PNAPipeline(Pipeline):
    DEVICE = "cpu"
    PNA_AGGREGATORS = ["mean", "min", "max", "std"]
    PNA_SCALERS = ["identity", "amplification", "attenuation"]

    def __init__(
        self,
        num_epochs: int = 120,
        batch_size: int = 32,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 5e-4,
        lr_step_size: int = 10,
        lr_gamma: float = 1.0,
        augmenter_kind: str = None,
        smiles: List[str] = None,
    ):
        self._transformer = PYGGraphTransformer(atom_featurizer=AtomCalculator(), bond_featurizer=EdgeMatCalculator())
        self._transformer.auto_self_loop()
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._hidden_channels = hidden_channels
        self._num_layers = num_layers
        self._dropout = dropout
        self._learning_rate = learning_rate
        self._lr_step_size = lr_step_size
        self._lr_gamma = lr_gamma
        self._augmenter = augmenter_factory(augmenter_kind, self._transformer, smiles)
        self._model = None

    def transform_smiles(self, smiles: np.array) -> List[Data]:
        transformed_mols = self._transformer(smiles)
        # Dynamically attach the smile to each transformed molecule (needed by one of the augmenters)
        for mol, smile in zip(transformed_mols, smiles):
            mol.smile = smile
        return transformed_mols

    def split_data(
        self, X: List[Data], y: List[int], train_indices: np.array, test_indices: np.array
    ) -> Tuple[List[Data], List[Data], List[int], List[int]]:
        X_train = [X[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]
        y_train = [y[i] for i in train_indices]
        y_test = [y[i] for i in test_indices]
        return X_train, X_test, y_train, y_test

    @property
    def num_atom_features(self) -> int:
        return self._transformer.atom_dim

    @property
    def num_bond_features(self) -> int:
        return self._transformer.bond_dim

    def _get_model(self, X: List[Data] = None) -> PNA:
        if self._model is None:
            self._model = PNA(
                in_channels=self.num_atom_features,
                hidden_channels=self._hidden_channels,
                num_layers=self._num_layers,
                out_channels=1,
                dropout=self._dropout,
                act="relu",
                edge_dim=self.num_bond_features,
                aggregators=self.PNA_AGGREGATORS,
                scalers=self.PNA_SCALERS,
                deg=self._get_degree(X),
            )
        return self._model

    def _get_degree(self, X: List[Data]) -> torch.Tensor:
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

    def forward(self, data: Data) -> torch.Tensor:
        out = self._model(data.x, data.edge_index, edge_attr=data.edge_attr)
        out = global_add_pool(out, data.batch)
        return F.sigmoid(out)

    def _compute_auc(self, y_hat: torch.Tensor, y_true: torch.Tensor) -> float:
        y_hat = torch.cat(y_hat).numpy()
        y_true = torch.cat(y_true).numpy()
        return roc_auc_score(y_true, y_hat)

    def _train_one_epoch(
        self, loss_fn: BCELoss, model: PNA, optimizer: torch.optim.Adam, train_loader: DataLoader
    ) -> Tuple[float, List[float], List[int]]:
        model.train()
        losses = []
        y_hat = []
        y_true = []
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
        return np.mean(losses), y_hat, y_true

    def _eval_one_epoch(
        self, loss_fn: BCELoss, model: PNA, test_loader: DataLoader
    ) -> Tuple[float, List[float], List[int]]:
        model.eval()
        losses = []
        test_y_hat = []
        test_y_true = []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.DEVICE)
                out = model(data.x, data.edge_index, edge_attr=data.edge_attr)
                out = global_add_pool(out, data.batch)
                out = F.sigmoid(out)
                loss = loss_fn(out.squeeze(), data.y)
                losses.append(loss.item())
                test_y_hat.append(out.detach().cpu().squeeze())
                test_y_true.append(data.y)
        return np.mean(losses), test_y_hat, test_y_true

    def fit(
        self, X_train: List[Data], y_train: List[int], X_test: List[Data] = None, y_test: List[int] = None
    ) -> Dict[str, List[float]]:
        train_loader = build_dataset(
            self._transformer, X_train, y_train, self._augmenter, shuffle=True, batch_size=self._batch_size
        )
        test_loader = None
        if X_test is not None:
            test_loader = build_dataset(self._transformer, X_test, y_test, shuffle=False, batch_size=self._batch_size)

        train_aucs = []
        test_aucs = []
        loss_fn = BCELoss()
        model = self._get_model(X_train)
        optimizer = torch.optim.Adam(model.parameters(), lr=self._learning_rate)
        lr_scheduler = StepLR(optimizer, step_size=self._lr_step_size, gamma=self._lr_gamma)
        with tqdm(range(self._num_epochs)) as pbar:
            for epoch in pbar:
                # Perform one epoch on training data
                mean_train_loss, train_y_hat, train_y_true = self._train_one_epoch(
                    loss_fn, model, optimizer, train_loader
                )

                lr_scheduler.step()

                # Compute training metrics
                train_auc = self._compute_auc(train_y_hat, train_y_true)
                train_aucs.append(train_auc)

                if test_loader is not None:
                    mean_test_loss, test_y_hat, test_y_true = self._eval_one_epoch(loss_fn, model, test_loader)

                    # Compute test metrics
                    test_auc = self._compute_auc(test_y_hat, test_y_true)
                    test_aucs.append(test_auc)

                    pbar.set_description(
                        f"Epoch {epoch} - Train Loss {mean_train_loss:.3f} - Test Loss {mean_test_loss:.3f}"
                        f"- AUC Train {train_auc:.3f} - AUC Test {test_auc:.3f} "
                    )
                else:
                    pbar.set_description(
                        f"Epoch {epoch} - Train Loss {mean_train_loss:.3f} - AUC Train {train_auc:.3f} "
                    )
        return {"train_aucs": train_aucs, "test_aucs": test_aucs}

    def predict_proba(self, X_test: List[Data], y_test: List[int]) -> np.array:
        test_loader = build_dataset(self._transformer, X_test, y_test, shuffle=False)
        model = self._get_model()
        model.eval()
        preds = []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.DEVICE)
                out = self.forward(data)
                preds.append(out.detach().cpu().squeeze())
        return torch.cat(preds).numpy()
