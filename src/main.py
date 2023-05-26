import logging
import os
import pickle
import warnings

import datamol as dm

from data import load_and_prepare_dataset, preprocess_smiles
from pipeline_factory import get_pipelines
from training import cross_validate


def silence_warnings():
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    dm.disable_rdkit_log()


def main():
    smiles, y = load_and_prepare_dataset()
    smiles = preprocess_smiles(smiles)
    transformers = get_pipelines(smiles)
    cv_results = cross_validate(smiles, y, transformers)
    logging.info("Cross validation performed")
    logging.info(f"Best model AUC: {cv_results.best_auc}")
    logging.info(f"Best pipeline: {cv_results.best_pipeline}")
    pickle.dump(cv_results, open("cv_results.df", "wb"))
    logging.info("Results saved on disk")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    silence_warnings()
    main()
