import itertools
import logging
from collections import defaultdict
from typing import Dict, List

import datamol as dm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from pipeline import Pipeline


class TrainingResult:
    def __init__(self, results_df: pd.DataFrame, best_auc: float, best_y_test: np.array, best_y_score: np.array):
        self.results_df = results_df
        self.best_auc = best_auc
        self.best_y_test = best_y_test
        self.best_y_score = best_y_score

    @property
    def best_transformer(self):
        mean_test_aucs = self.results_df["Mean Test AUC"]
        best_row = self.results_df[mean_test_aucs == mean_test_aucs.max()]
        return best_row.index.values[0]


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
    return GroupShuffleSplit(n_splits=3, test_size=0.2, random_state=42).split(smiles, groups=scaffolds)


def _add_mean_test_auc(results) -> pd.DataFrame:
    results_df = pd.DataFrame(results)
    mean_aucs = results_df.groupby("Transformer")["Test AUC"].mean()
    mean_aucs_df = pd.DataFrame({"Mean Test AUC": mean_aucs})
    return results_df.set_index("Transformer").join(mean_aucs_df)


def cross_validate(smiles: np.array, y: np.array, pipelines: Dict[str, Pipeline]) -> TrainingResult:
    """
    Assess the best transformer to use for the prediction with scaffold cross validation.
    :param pipelines: Dictionary of pipelines that we want to assess for molecule featurization + modeling
    :param smiles: List of modules in their smile string representation
    :param y: The target we want to predict (pIC50 > 8)
    :return: A TrainingResult instance from which we can extract the metric of interest (MAE in this case)
    for each fold along with the best model and best metric value
    """
    scaffolds = _get_scaffolds(smiles)
    results = defaultdict(list)
    best_auc = 0.0
    for name, pipeline in pipelines.items():
        # Transform features
        logging.info(f"Transforming features with transformer: {name}")
        X = pipeline.transform_smiles(smiles)
        splitter = _get_cv_splitter(scaffolds, smiles)
        aucs = []
        all_y_test = []
        all_y_score = []
        for i_fold, (train_indices, test_indices) in enumerate(splitter):
            # Split data
            X_train, X_test, y_train, y_test = pipeline.split_data(X, y, train_indices, test_indices)

            # Fit model
            pipeline.fit(X_train, y_train)

            # Predict
            y_score = pipeline.predict_proba(X_test, y_test)

            # Compute metrics
            auc = roc_auc_score(y_test, y_score)
            aucs.append(auc)
            logging.info(f"Fold {i_fold} - Test AUC: {auc:.3f}")

            # Save metrics
            results["Transformer"].append(name)
            results["Fold"].append(i_fold)
            results["Test AUC"].append(auc)
            all_y_test.append(y_test)
            all_y_score.append(y_score)

        mean_auc = np.mean(aucs)
        logging.info(f"Mean Test AUC: {mean_auc:.3f}")
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_y_test = list(itertools.chain.from_iterable(all_y_test))
            best_y_score = list(itertools.chain.from_iterable(all_y_score))

    results_df = _add_mean_test_auc(results)
    return TrainingResult(results_df, best_auc, best_y_test, best_y_score)
