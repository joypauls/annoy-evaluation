"""Builds the search indexes for each dataset and each candidate library."""
import numpy as np
from annoy import AnnoyIndex
import faiss
import pickle
import logging
from time import time

from src.config import BUILT_INDEX_DIR, PROCESSED_PATHS, DATASET_IDS, N_LIST, N_TREES
from src.dataset import BenchmarkDataset, get_dataset
from src.annoy import Annoy

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s :: %(levelname)s :: %(message)s"
)


def get_dataset(path: str) -> BenchmarkDataset:
    """
    Loads a BenchmarkDataset object from a pickle file.
    """
    with open(path, "rb") as f:
        bd = pickle.load(f)
    return bd


if __name__ == "__main__":
    logging.info("Begin building search indexes")
    start = time()

    # iterate over datasets
    for i, processed_path in enumerate(PROCESSED_PATHS):
        # load dataset
        logging.info("=> BUILDING INDEXES FOR {}".format(processed_path))
        bd = get_dataset(processed_path)
        embedding_dim = bd.train.shape[1]

        # build annoy
        annoy_index = Annoy(dim=embedding_dim, name="{}_annoy".format(DATASET_IDS[i]))
        annoy_index.build(bd.train, N_TREES)
        annoy_index.save(BUILT_INDEX_DIR)
        logging.info("Built: {}".format(annoy_index.name))

    end = time()

    logging.info("Done in {:.5f} s".format(end - start))
