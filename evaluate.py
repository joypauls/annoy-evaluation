"""Evaluate methods over the test sets."""
import numpy as np
from annoy import AnnoyIndex
import faiss
import pickle
import logging
from time import time

from src.config import BUILT_INDEX_DIR, K_NEIGHBORS_OVERALL, PROCESSED_PATHS, DATASET_IDS, K_NEIGHBORS_MAX
from src.dataset import BenchmarkDataset, get_dataset
from src.annoy import Annoy
from src.plotting import recall_distribution_plot
from src.metrics import average_recall

logging.basicConfig(level=logging.INFO, format="%(asctime)s :: %(levelname)s :: %(message)s")


if __name__ == "__main__":
    start = time()

    # iterate over datasets
    for i, processed_path in enumerate(PROCESSED_PATHS[0:1]):
        # load dataset
        logging.info("=> EVALUATING RESULTS FOR {}".format(processed_path))
        bd = get_dataset(processed_path)

        # load results
        annoy_results = np.load("./data/annoy_results_{}.npy".format(DATASET_IDS[i]))
        faiss_results = np.load("./data/faiss_results_{}.npy".format(DATASET_IDS[i]))

        print(faiss_results[0:10, 0:10])

        # initialize recall distribution arrays
        annoy_recall_distribution = np.zeros((annoy_results.shape[0],), dtype=np.float32)
        faiss_recall_distribution = np.zeros((faiss_results.shape[0],), dtype=np.float32)

        # iterate to do calculations
        # TODO: this can be more efficient
        for j, x in enumerate(annoy_results):
          annoy_recall_distribution[j] = average_recall(
            bd.ground_truth[j, 0:K_NEIGHBORS_OVERALL], 
            x[0:K_NEIGHBORS_OVERALL], 
            K_NEIGHBORS_OVERALL
          )
          faiss_recall_distribution[j] = average_recall(
            bd.ground_truth[j, 0:K_NEIGHBORS_OVERALL], 
            faiss_results[j, 0:K_NEIGHBORS_OVERALL], 
            K_NEIGHBORS_OVERALL
          )

        print(faiss_recall_distribution[0:10])

        recall_distribution_plot(annoy_recall_distribution, faiss_recall_distribution, DATASET_IDS[i])
        logging.info("Annoy MAR: {}, Faiss MAR: {}".format(np.mean(annoy_recall_distribution), np.mean(faiss_recall_distribution)))






        # prep
        # num_test_vectors = bd.test.shape[0]

        # can precompute in a batch
        # _, faiss_estimated = faiss_index.search(glove25_test_np_norm[0:100], k)

        # annoy_neighbors = np.zeros((num_test_vectors, K_NEIGHBORS_MAX), dtype=np.int32)
        # faiss_neighbors = np.zeros((test_size, k), dtype=np.int32)
        # results_summary = []

        # # annoy
        # annoy_index = Annoy(
        #     dim=embedding_dim,
        #     metric="angular",
        #     name="{}_annoy".format(DATASET_IDS[i])
        # )
        # annoy_index.load_from_file(BUILT_INDEX_DIR + DATASET_IDS[i] + "_annoy_index")
        # annoy_neighbors = annoy_index.batch_query(bd.test, K_NEIGHBORS_MAX)
        # np.save("./data/annoy_results_{}.npy".format(DATASET_IDS[i]), annoy_neighbors)  # save the array
        # logging.info("{} results: {}".format(annoy_index.name, annoy_neighbors.shape))

        # build faiss
        # faiss_index = Annoy(
        #     dim=embedding_dim,
        #     metric="angular",
        #     name="{}_annoy".format(DATASET_IDS[i])
        # )
        # annoy_index.load_from_file(BUILT_INDEX_DIR + DATASET_IDS[i] + "_annoy_index")
        # annoy_neighbors = annoy_index.batch_query(bd.test, K_NEIGHBORS_MAX)
        # np.save("./data/annoy_results_{}.npy".format(DATASET_IDS[i]), annoy_neighbors)  # save the array
        # logging.info("{} results: {}".format(annoy_index.name, annoy_neighbors.shape))

    end = time()

    logging.info("Done in {:.5f} s".format(end - start))
