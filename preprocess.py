"""This script converts the dataset from its original form to a more convenient format."""
import csv
import logging
import pickle
from time import time

from src.dataset import BenchmarkDataset
from src.config import RAW_PATHS, PROCESSED_PATHS, DATASET_IDS, K_NEIGHBORS_MAX

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s :: %(levelname)s :: %(message)s"
)


def save_dataset(d: BenchmarkDataset, path: str):
    with open(path, "wb") as f:
        pickle.dump(d, f)


if __name__ == "__main__":
    logging.info("Begin preprocessing datasets")

    start = time()

    for i, raw_path in enumerate(RAW_PATHS):
        logging.info("=> PROCESSING {}".format(raw_path))
        bd = BenchmarkDataset(id=DATASET_IDS[i])
        bd.load_from_h5py(raw_path)
        logging.info(bd)
        bd.compute_neighbors(metric="angular", k=K_NEIGHBORS_MAX)
        logging.info(bd)
        save_dataset(bd, PROCESSED_PATHS[i])

    end = time()

    logging.info("Done in {:.5f} s".format(end - start))
