"""Evaluate methods over the test sets."""
import numpy as np
import pandas as pd
import pickle
import logging
from time import time

from src.config import BUILT_INDEX_DIR, K_NEIGHBORS_OVERALL, PROCESSED_PATHS, DATASET_IDS, K_NEIGHBORS_MAX
from src.dataset import BenchmarkDataset, get_dataset
from src.annoy import Annoy
from src.plotting import recall_distribution_plot, recall_curve_plot
from src.metrics import average_recall

logging.basicConfig(level=logging.INFO, format="%(asctime)s :: %(levelname)s :: %(message)s")


if __name__ == "__main__":
    start = time()

    # gather results over all datasets then plot
    recall_distributions = []
    # recall_depth_results = []

    # iterate over datasets
    # TODO: i'm sure a ton of these operations can be vectorized
    for i, processed_path in enumerate(PROCESSED_PATHS):
        # load dataset
        logging.info("=> EVALUATING RESULTS FOR {}".format(processed_path))
        bd = get_dataset(processed_path)

        # load results
        annoy_results = np.load("./data/annoy_results_{}.npy".format(DATASET_IDS[i]))

        # initialize recall distribution array
        recall_distribution = np.zeros((annoy_results.shape[0],), dtype=np.float32)
        
        # # initialize recall depth array
        # recall_depth = np.zeros((100, K_NEIGHBORS_MAX), dtype=np.float32)
        # # recall_depth = np.zeros((annoy_results.shape[0], K_NEIGHBORS_MAX), dtype=np.float32)

        # iterate to do calculations
        # TODO: this can be more efficient
        for j, x in enumerate(annoy_results):
          # recall distribution at a single search depth
          recall_distribution[j] = average_recall(
            bd.ground_truth[j, 0:K_NEIGHBORS_OVERALL], 
            x[0:K_NEIGHBORS_OVERALL], 
            K_NEIGHBORS_OVERALL
          )

          # cur_query_recall = np.zeros((K_NEIGHBORS_MAX,), dtype=np.float32)
          # # recall calculated at each depth
          # for d in range(1, K_NEIGHBORS_MAX+1):
          #   cur_query_recall[d-1] = average_recall(
          #     bd.ground_truth[j, 0:d], 
          #     x[0:d], 
          #     d
          #   )

          # # assign to matrix of results
          # recall_depth[j] = cur_query_recall

        # gather results of the whole dataset
        recall_distributions.append(recall_distribution)
        # recall_depth_results.append(np.apply_along_axis(np.mean, 0, recall_depth))
        logging.info("Annoy MAR ({}): {}".format(DATASET_IDS[i], np.mean(recall_distribution)))

    recall_distribution_plot(recall_distributions, DATASET_IDS)
    # recall_curve_plot(recall_depth_results, DATASET_IDS)

    end = time()

    logging.info("Done in {:.5f} s".format(end - start))
