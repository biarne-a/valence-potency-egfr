import numpy as np


class Pipeline:
    def transform_smiles(self, smiles: np.array) -> np.array:
        """
        Takes as input the smile strings and apply a given molecule transformer.
        The result is a numpy array of dimension (number of molecules, dimension of a featurized module)
        """
        raise NotImplementedError

    def fit(self, X_train: np.array, y_train: np.array):
        raise NotImplementedError

    def predict_proba(self, X_test: np.array, y_test: np.array):
        raise NotImplementedError

    def split_data(self, X, y, train_indices, test_indices):
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
