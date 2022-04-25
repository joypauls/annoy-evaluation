"""Evaluate methods over the test sets."""
import numpy as np
from annoy import AnnoyIndex
import faiss
import pickle
import logging
from time import time

from src.config import BUILT_INDEX_DIR, PROCESSED_PATHS, DATASET_IDS, K_NEIGHBORS_MAX
from src.dataset import BenchmarkDataset, get_dataset
from src.annoy import Annoy
from src.faiss import Faiss

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s :: %(levelname)s :: %(message)s")


if __name__ == "__main__":
    start = time()
    # iterate over datasets
    for i, processed_path in enumerate(PROCESSED_PATHS):
        # load dataset
        logging.info("=> GENERATING RESULTS FOR {}".format(processed_path))
        bd = get_dataset(processed_path)
        embedding_dim = bd.train.shape[1]

        # prep
        num_test_vectors = bd.test.shape[0]

        # can precompute in a batch
        # _, faiss_estimated = faiss_index.search(glove25_test_np_norm[0:100], k)

        # annoy_neighbors = np.zeros((num_test_vectors, K_NEIGHBORS_MAX), dtype=np.int32)
        # faiss_neighbors = np.zeros((test_size, k), dtype=np.int32)
        # results_summary = []

        # annoy results
        annoy_index = Annoy(
            dim=embedding_dim,
            name="{}_annoy".format(DATASET_IDS[i])
        )
        annoy_index.load_from_file(BUILT_INDEX_DIR + DATASET_IDS[i] + "_annoy_index")
        annoy_neighbors = annoy_index.batch_query(bd.test, K_NEIGHBORS_MAX)
        np.save("./data/annoy_results_{}.npy".format(DATASET_IDS[i]), annoy_neighbors)  # save the array
        logging.info("{} results: {}".format(annoy_index.name, annoy_neighbors.shape))

        # faiss results
        faiss_index = Faiss(
            dim=embedding_dim,
            name="{}_faiss".format(DATASET_IDS[i])
        )
        faiss_index.load_from_file(BUILT_INDEX_DIR + DATASET_IDS[i] + "_faiss_index")
        faiss_neighbors = faiss_index.batch_query(bd.test, K_NEIGHBORS_MAX)

        print(faiss_index.query(bd.test[0], K_NEIGHBORS_MAX))
        print(faiss_neighbors[0:10, 0:10])

        np.save("./data/faiss_results_{}.npy".format(DATASET_IDS[i]), faiss_neighbors)  # save the array
        logging.info("{} results: {}".format(faiss_index.name, faiss_neighbors.shape))

    end = time()

    logging.info("Done in {:.5f} s".format(end - start))
