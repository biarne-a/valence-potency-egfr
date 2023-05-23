import logging
from typing import Optional, Tuple

import datamol as dm
import numpy as np
import pandas as pd
from rdkit.Chem import SaltRemover

CSV_FILE = "data/EGFR_compounds_lipinski.csv"


def load_and_prepare_dataset() -> Tuple[np.array, np.array]:
    """
    Loads the EGFR dataset and prepare training data.
    We want to determine if a molecule has potency value (pIC50) greater than 8.
    :return: Both the molecules SMILES strings and the associated label
    """
    logging.info("Loading the EGFR dataset")
    df = pd.read_csv(CSV_FILE)
    smiles = df["smiles"].values
    y = (df["pIC50"].values > 8.0).astype(int)
    return smiles, y


def preprocess_smiles(smiles: np.array) -> np.array:
    logging.info("Preprocessing the SMILES strings")
    smiles = np.array([_preprocess_smile(smi) for smi in smiles])
    return np.array([smi for smi in smiles if dm.to_mol(smi) is not None])


def _preprocess_smile(smi: str) -> Optional[str]:
    """Preprocesses one SMILE string"""
    mol = dm.to_mol(smi, ordered=True, sanitize=False)
    try:
        mol = dm.sanitize_mol(mol)
    except Exception:
        mol = None

    if mol is None:
        return None

    mol = dm.standardize_mol(mol, disconnect_metals=True)
    remover = SaltRemover.SaltRemover()
    mol = remover.StripMol(mol, dontRemoveEverything=True)

    return dm.to_smiles(mol)
