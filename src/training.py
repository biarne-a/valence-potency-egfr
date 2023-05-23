import logging
from collections import defaultdict
from typing import Dict, List

import datamol as dm
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from molfeat.trans import MoleculeTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


class TrainingResult:
    def __init__(self, results_df: pd.DataFrame, best_model: CatBoostClassifier, best_auc: float):
        self.results_df = results_df
        self.best_model = best_model
        self.best_auc = best_auc

    @property
    def best_transformer(self):
        test_aucs = self.results_df["Test AUC"]
        best_row = self.results_df[test_aucs == test_aucs.max()]
        return best_row["Transformer"].values[0]


def _get_scaffolds(smiles) -> List[str]:
    """
    Returns the foundational molecular structures (scaffolds) associated with each input smile string.
    To be used in order to group the smiles by scaffold and perform cross validation by group
    """
    return [dm.to_smiles(dm.to_scaffold_murcko(dm.to_mol(smi))) for smi in smiles]


def _get_cv_splitter(scaffolds: List[str], smiles: np.array):
    """
    Returns a splitter to be used for cross validation.
    A splitter splits the data into a training set and a test set.
    Takes as input the group identifiers (scaffolds) per smile and perform the splits in accordance to the groups.
    We want to ensure that the test set is made of observations with different scaffolds in order to evaluate
    performance under more realistic screening settings (where the model will be applied on molecules with scaffolds
    unseen during training).
    """
    return GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42).split(smiles, groups=scaffolds)


def _transform_smiles(smiles: np.array, transformer: MoleculeTransformer) -> np.array:
    """
    Takes as input the smile strings and apply a given molecule transformer.
    The result is a numpy array of dimension (number of molecules, dimension of a featurized module)
    """
    features, indices = transformer(smiles, ignore_errors=True)
    return np.array(features)[indices]


def _new_regression_model() -> CatBoostClassifier:
    return CatBoostClassifier(iterations=3000, depth=6, loss_function="Logloss", verbose=False)


def cross_validate(smiles: np.array, y: np.array, transformers: Dict[str, MoleculeTransformer]) -> TrainingResult:
    """
    Assess the best transformer to use for the prediction with scaffold cross validation.
    :param transformers: Dictionary of transformers that we want to assess for molecule featurization
    :param smiles: List of modules in their smile string representation
    :param y: The target we want to predict (pIC50 > 8)
    :return: A TrainingResult instance from which we can extract the metric of interest (MAE in this case)
    for each fold along with the best model and best metric value
    """
    scaffolds = _get_scaffolds(smiles)
    results = defaultdict(list)
    best_model = None
    best_auc = 0.0
    for name, transformer in transformers.items():
        # Transform features
        logging.info(f"Transforming features with transformer: {name}")
        X = _transform_smiles(smiles, transformer)
        splitter = _get_cv_splitter(scaffolds, smiles)
        aucs = []
        for i_fold, (train_indices, test_indices) in enumerate(splitter):
            # Split data
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            # Fit model
            model = _new_regression_model()
            model.fit(X_train, y_train)

            # Predict
            y_score = model.predict_proba(X_test)

            # Compute metrics
            auc = roc_auc_score(y_test, y_score[:, 1])
            aucs.append(auc)
            logging.info(f"Fold {i_fold} - Test AUC: {auc}")

            results["Transformer"].append(name)
            results["Fold"].append(i_fold)
            results["Test AUC"].append(auc)

        mean_auc = np.mean(aucs)
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_model = model

    results_df = pd.DataFrame(results)
    return TrainingResult(results_df, best_model, best_auc)
