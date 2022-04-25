"""Evaluate methods over the test sets."""
import numpy as np
import logging
from time import time

from src.config import (
    K_NEIGHBORS_OVERALL,
    PROCESSED_PATHS,
    DATASET_IDS,
    K_NEIGHBORS_MAX,
)
from src.dataset import get_dataset
from src.plotting import recall_distribution_plot
from src.metrics import average_recall

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s :: %(levelname)s :: %(message)s"
)


if __name__ == "__main__":
    start = time()
    # list to gather results over all datasets
    recall_distributions = []

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

        # iterate to do calculations
        # TODO: this can be more efficient
        for j, x in enumerate(annoy_results):
            # recall distribution at a single search depth
            recall_distribution[j] = average_recall(
                bd.ground_truth[j, 0:K_NEIGHBORS_OVERALL],
                x[0:K_NEIGHBORS_OVERALL],
                K_NEIGHBORS_OVERALL,
            )

        # gather results of the whole dataset
        recall_distributions.append(recall_distribution)
        logging.info(
            "Annoy MAR ({}): {}".format(DATASET_IDS[i], np.mean(recall_distribution))
        )

    recall_distribution_plot(recall_distributions, DATASET_IDS)

    end = time()

    logging.info("Done in {:.5f} s".format(end - start))
