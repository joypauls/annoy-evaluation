"""Plotting helper functions."""
import numpy as np
import matplotlib.pyplot as plt


def recall_distribution_plot(annoy_recall: np.ndarray, faiss_recall: np.ndarray, dataset_id: str):
  """
  Plots the distribution of average recall for the test set.
  """
  bins = 20
  annoy_mean = np.mean(annoy_recall)
  annoy_plot_weights = np.ones(len(annoy_recall)) / len(annoy_recall)
  faiss_mean = np.mean(faiss_recall)
  faiss_plot_weights = np.ones(len(faiss_recall)) / len(faiss_recall)


  n1, _, _ = plt.hist(annoy_recall, weights=annoy_plot_weights, bins=bins, alpha=0.5, label="Annoy")
  plt.axvline(annoy_mean, color="#4c72b0", alpha=0.6, ls="--", label="Annoy MAR@100")

  n2, _, _ = plt.hist(faiss_recall, weights=faiss_plot_weights, bins=bins, alpha=0.5, label="Faiss")
  plt.axvline(faiss_mean, color="#cc642a", alpha=0.6, ls="--", label="Faiss MAR@100")

  plot_height = max(np.max(n1), np.max(n2))  # in y-axis units
  plt.text(annoy_mean + 0.01, plot_height*0.8, "MAR = {:.2f}".format(annoy_mean), rotation=90, verticalalignment="center")
  plt.text(faiss_mean + 0.01, plot_height*0.8, "MAR = {:.2f}".format(faiss_mean), rotation=90, verticalalignment="center")

  plt.legend()
  plt.title("Distribution of Recall@100 for Test Queries ({})".format(dataset_id))
  plt.xlabel("Average Recall @ 100")
  plt.ylabel("Percent")

  plt.savefig("./plots/recall_distribution_{}.png".format(dataset_id))
