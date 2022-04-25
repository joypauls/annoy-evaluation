"""This script converts the dataset from it's original form (u.data) to a more convenient format."""
import csv
import logging
import pickle
from time import time

from src.dataset import BenchmarkDataset

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s :: %(levelname)s :: %(message)s")

# parameters
K_NEIGHBORS = 300

# paths to datasets
raw_paths = ["./data/raw/glove-25-angular.hdf5", "./data/raw/glove-50-angular.hdf5", "./data/raw/glove-100-angular.hdf5"]
output_paths = ["./data/glove-25-angular.pkl", "./data/glove-50-angular.pkl", "./data/glove-100-angular.pkl"]


def save_dataset(d: BenchmarkDataset, path: str):
    with open(path, "wb") as f:
        pickle.dump(d, f)


if __name__ == "__main__":
    logging.info("Begin preprocessing datasets")

    start = time()

    for i, raw_path in enumerate(raw_paths):
        logging.info("=> PROCESSING {}".format(raw_path))
        bd = BenchmarkDataset()
        bd.load_from_h5py(raw_path)
        logging.info(bd)
        bd.compute_neighbors(metric="angular", k=K_NEIGHBORS)
        logging.info(bd)
        save_dataset(bd, output_paths[i])

    end = time()

    logging.info("Done in {:.5f} s".format(end - start))
