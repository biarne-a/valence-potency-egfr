from typing import Dict

from molfeat.trans import FPVecTransformer, MoleculeTransformer
from molfeat.trans.pretrained.dgl_pretrained import PretrainedDGLTransformer
from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer


def _new_vec_transformer(kind: str) -> FPVecTransformer:
    return FPVecTransformer(kind, n_jobs=-1)


def _new_hf_tranformer(kind) -> PretrainedHFTransformer:
    return PretrainedHFTransformer(kind, notation="smiles")


def _new_dgl_tranformer(kind) -> PretrainedDGLTransformer:
    return PretrainedDGLTransformer(kind=kind, dtype=float)


def get_transformers() -> Dict[str, MoleculeTransformer]:
    return {
        "ecfp:4": _new_vec_transformer("ecfp:4"),
        "fcfp:4": _new_vec_transformer(kind="fcfp:4"),
        "mordred": _new_vec_transformer(kind="mordred"),
        "ChemBERTa-77M-MLM": _new_hf_tranformer("ChemBERTa-77M-MLM"),
        "gin_supervised_masking": _new_dgl_tranformer("gin_supervised_masking"),
        "gin_supervised_infomax": _new_dgl_tranformer("gin_supervised_infomax"),
        "gin_supervised_edgepred": _new_dgl_tranformer("gin_supervised_edgepred"),
    }
