"""Plotting helper functions."""
import numpy as np
import matplotlib.pyplot as plt
from typing import List

plt.rcParams["figure.figsize"] = (10, 6)

LINE_COLOR_SEQUENCE = ["#4c72b0", "#cc642a", "#0a610a"]


def recall_distribution_plot(recall: List[np.ndarray], dataset_ids: List[str]):
  """
  Plots the distribution of average recall for each dataset.
  """
  bins = 20
  means = [np.mean(x) for x in recall]
  plot_weights = np.ones(len(recall[0])) / len(recall[0])

  bar_n_list = []  # used for bar chart scaling stuff
  for i, x in enumerate(recall):
    n, _, _ = plt.hist(x, weights=plot_weights, bins=bins, alpha=0.5, label=dataset_ids[i])
    bar_n_list.append(n)

  plot_height = np.max(bar_n_list)
  for j, mean in enumerate(means):
    plt.axvline(mean, color=LINE_COLOR_SEQUENCE[j], alpha=0.6, ls="--", label="MAR@100 ({})".format(dataset_ids[j]))
    plt.text(mean + 0.01, plot_height*0.8, "MAR = {:.2f}".format(mean), rotation=90, verticalalignment="center")

  plt.legend()
  plt.title("Distribution of Annoy Recall@100 for Test Queries")
  plt.xlabel("Average Recall@100")
  plt.ylabel("Percent")

  plt.savefig("./plots/recall_distribution.png")
  plt.close()


def recall_curve_plot(recall: List[np.ndarray], dataset_ids: List[str]):
  """
  Plots the recall curve for each dataset.
  """
  depths = np.arange(1, recall[0].shape[0] + 1, 1, dtype=int)
  for i, x in enumerate(recall):
    plt.plot(depths, x)

  plt.savefig("./plots/recall_at_varying_depths.png")
  plt.close()
