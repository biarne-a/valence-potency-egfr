from typing import Dict, List, Optional

import numpy as np


class Pipeline:
    """
    A pipeline is an abstraction that tells us different things like:
    * How to preprocess (featurize) the data before feading it to our ML model
    * How to split the data into a training set and a test set
    * How to train and predict
    We have different versions of pipelines to handle different types of models
    """

    def transform_smiles(self, smiles: np.array) -> np.array:
        """
        Takes as input the smile strings and apply a given molecule transformer.
        The result is a numpy array of dimension (number of molecules, dimension of a featurized module)
        """
        raise NotImplementedError

    def fit(self, X_train: np.array, y_train: np.array, X_test=None, y_test=None) -> Optional[Dict[str, List[float]]]:
        """
        Learn by fitting the training data. Eventually also predict on the test set.
        """
        raise NotImplementedError

    def predict_proba(self, X_test: np.array, y_test: np.array):
        """
        Predict on the test set and return the associated probabilities
        """
        raise NotImplementedError

    def split_data(self, X, y, train_indices, test_indices):
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
