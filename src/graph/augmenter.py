import math
import random
from copy import deepcopy
from typing import Dict, List, Optional

import torch
from molfeat.trans.graph.adj import PYGGraphTransformer
from rdkit import Chem, DataStructs
from rdkit.Chem.BRICS import BRICSDecompose
from torch_geometric.data.data import Data
from tqdm import tqdm


class Augmenter:
    def augment(self, data: Data) -> Data:
        raise NotImplementedError


class RndAtomMaskAugmenter:
    NB_UNIQUE_ATOMS = 43

    def __init__(self, p: float = 0.2):
        self._p = p

    def augment(self, data: Data) -> Data:
        """Randomly mask an atom by clearing one hot encoded value"""
        N = data.x.size(0)
        num_mask_nodes = max([1, math.floor(self._p * N)])
        mask_nodes = random.sample(list(range(N)), num_mask_nodes)

        aug_data = deepcopy(data)
        for atom_idx in mask_nodes:
            aug_data.x[atom_idx, : self.NB_UNIQUE_ATOMS] = torch.tensor([0] * self.NB_UNIQUE_ATOMS)
        return aug_data


class RndBondDeleteAugmenter:
    def __init__(self, p: float = 0.2):
        self._p = p

    def augment(self, data: Data) -> Data:
        """
        Randomly delete chemical bonds given a certain ratio
        """
        M = data.edge_index.size(1) // 2
        num_mask_edges = max([0, math.floor(self._p * M)])
        mask_edges_single = random.sample(list(range(M)), num_mask_edges)
        mask_edges = [2 * i for i in mask_edges_single] + [2 * i + 1 for i in mask_edges_single]

        aug_data = deepcopy(data)
        dim_edges_attr = data.edge_attr.size(1)
        aug_data.edge_index = torch.zeros((2, 2 * (M - num_mask_edges)), dtype=torch.int64)
        aug_data.edge_attr = torch.zeros((2 * (M - num_mask_edges), dim_edges_attr), dtype=torch.float32)
        count = 0
        for bond_idx in range(2 * M):
            if bond_idx not in mask_edges:
                aug_data.edge_index[:, count] = data.edge_index[:, bond_idx]
                aug_data.edge_attr[count, :] = data.edge_attr[bond_idx, :]
                count += 1
        return aug_data


class RndMotifRemovalAugmenter:
    def __init__(self, transformer: PYGGraphTransformer, smiles: List[str], similarity_threshold: float = 0.6):
        self._transformer = transformer
        self._sub_smiles_dict = self._build_subsmiles(smiles, similarity_threshold)

    def _build_subsmiles(self, smiles, similarity_threshold) -> Dict[str, List[str]]:
        result = {}
        for smile in tqdm(smiles):
            mol = Chem.MolFromSmiles(smile)
            sub_smiles = []
            fp = Chem.RDKFingerprint(mol)
            res = list(BRICSDecompose(mol, returnMols=False, singlePass=True))
            for r in res:
                sub_mol = Chem.MolFromSmiles(r)
                fp_aug = Chem.RDKFingerprint(sub_mol)
                if DataStructs.FingerprintSimilarity(fp, fp_aug) > similarity_threshold:
                    sub_smiles.append(r)
            result[smile] = sub_smiles
        return result

    def augment(self, data) -> Data:
        """
        Randomly removes a motif decomposed via BRICS
        """
        sub_smiles = self._sub_smiles_dict[data.smile]

        if len(sub_smiles):
            return data

        sub_smile = random.choice(sub_smiles)
        return self._transformer(sub_smile)


class ComposableAugmenter(Augmenter):
    def __init__(self, augmenters: List[Augmenter], p: float = 1.0):
        self._augmenters = augmenters
        self._p = p

    def augment(self, data: Data) -> Data:
        """
        Applies the list of augmenters in order to the molecule
        """
        if random.random() > self._p:
            return data

        for augmenter in self._augmenters:
            data = augmenter.augment(data)

        return data


class OneOfAugmenter(Augmenter):
    def __init__(self, augmenters: List[Augmenter], p: float = 1.0):
        self._p = p
        self._augmenters = augmenters

    def augment(self, data: Data) -> Data:
        """
        Applies one of the augmenters to the molecule
        """
        if random.random() > self._p:
            return data

        augmenter = random.choices(self._augmenters)[0]
        return augmenter.augment(data)


def augmenter_factory(
    kind: str, transformer: PYGGraphTransformer = None, smiles: List[str] = None
) -> Optional[Augmenter]:
    if kind == "rnd-mask":
        return RndAtomMaskAugmenter()
    if kind == "rnd-bond":
        return RndBondDeleteAugmenter()
    if kind == "rnd-struct":
        return RndMotifRemovalAugmenter(transformer, smiles)
    if kind == "rnd-comp":
        return ComposableAugmenter([RndAtomMaskAugmenter(), RndBondDeleteAugmenter()])
    if kind == "rnd-one-off":
        return OneOfAugmenter([RndAtomMaskAugmenter(), RndBondDeleteAugmenter()])
    return None
