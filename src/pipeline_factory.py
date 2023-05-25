from typing import Any, Dict

import numpy as np
from catboost import CatBoostClassifier
from molfeat.trans import FPVecTransformer, MoleculeTransformer
from molfeat.trans.pretrained.dgl_pretrained import PretrainedDGLTransformer
from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer

from graph.augmenter import (
    Augmenter,
    RndAtomMaskAugmenter,
)
from graph.pna_pipeline import PNAPipeline
from pipeline import Pipeline


class VecCatBoostPipeline(Pipeline):
    def __init__(self, mol_transformer: MoleculeTransformer, model: Any):
        self.mol_transformer = mol_transformer
        self.model = model

    def transform_smiles(self, smiles: np.array) -> np.array:
        """
        Takes as input the smile strings and apply a given molecule transformer.
        The result is a numpy array of dimension (number of molecules, dimension of a featurized module)
        """
        features, indices = self.mol_transformer(smiles, ignore_errors=True)
        return np.array(features)[indices]

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict_proba(self, X, y):
        y_score = self.model.predict_proba(X)
        pos_y_score = y_score[:, 1]
        return pos_y_score


def _new_catboost_model() -> CatBoostClassifier:
    return CatBoostClassifier(iterations=3000, depth=6, loss_function="Logloss", verbose=False)


def _new_vec_pipeline(kind: str) -> VecCatBoostPipeline:
    return VecCatBoostPipeline(mol_transformer=FPVecTransformer(kind, n_jobs=-1), model=_new_catboost_model())


def _new_hf_pipeline(kind) -> VecCatBoostPipeline:
    return VecCatBoostPipeline(
        mol_transformer=PretrainedHFTransformer(kind, notation="smiles"), model=_new_catboost_model()
    )


def _new_dgl_pipeline(kind) -> VecCatBoostPipeline:
    return VecCatBoostPipeline(
        mol_transformer=PretrainedDGLTransformer(kind=kind, dtype=float), model=_new_catboost_model()
    )


def _new_pna_pipeline(augmenter: Augmenter = None) -> PNAPipeline:
    return PNAPipeline(augmenter)


def get_pipelines() -> Dict[str, Pipeline]:
    return {
        "ecfp:4": _new_vec_pipeline("ecfp:4"),
        "fcfp:4": _new_vec_pipeline(kind="fcfp:4"),
        "mordred": _new_vec_pipeline(kind="mordred"),
        "ChemBERTa-77M-MLM": _new_hf_pipeline("ChemBERTa-77M-MLM"),
        "gin_supervised_masking": _new_dgl_pipeline("gin_supervised_masking"),
        "gin_supervised_infomax": _new_dgl_pipeline("gin_supervised_infomax"),
        "gin_supervised_edgepred": _new_dgl_pipeline("gin_supervised_edgepred"),
        "PNA": _new_pna_pipeline(),
        "PNA-rnd-mask-aug": _new_pna_pipeline(RndAtomMaskAugmenter()),
        "PNA-rnd-bond-aug": _new_pna_pipeline(RndBondDeleteAugmenter()),
        "PNA-rnd-struct-aug": _new_pna_pipeline(RndMotifRemovalAugmenter()),
        "PNA-rnd-comp-aug": _new_pna_pipeline([RndAtomMaskAugmenter(), RndBondDeleteAugmenter()]),
        "PNA-rnd-one-off-aug": _new_pna_pipeline([RndAtomMaskAugmenter(), RndBondDeleteAugmenter()]),
    }
