"""Annoy"""
import numpy as np
import annoy
from search_index import SearchIndex

SAVE_PATH = "../models/"


class Annoy(SearchIndex):
  """
  Description
  """
  def __init__(self, dim: int, metric = "angular", name: str = "annoy"):
    """
    Ideally the name passed includes context for the training data.
    """
    super().__init__(name=name)
    # initialize index
    self.index = annoy.AnnoyIndex(dim, metric)

  def build(self, training_data: np.ndarray, n_trees: int = 10):
    """
    """
    if len(training_data.shape) != 2:
      raise ValueError("training_data has the wrong shape.")

    try:
      _ = [self.index.add_item(i, v) for i, v in enumerate(training_data)]
      self.index.build(n_trees)
    except Exception as e:
      raise e

  def query(self, vector: np.ndarray, k: int) -> np.ndarray:
    """
    Returns k nearest neighbors by id.
    """
    if len(vector) != 1:
      raise ValueError("vector can only have shape (d,).")

    return self.index.get_nns_by_vector(vector, k)

  def batch_query(self, matrix: np.ndarray, k: int) -> np.ndarray:
    """
    Return nearest neighbors for each row vector.
    """
    # initialize the numpy array
    neighbors = np.zeros((matrix.shape[0],), dtype=np.float32)

    # annoy can't batch
    for i, x in enumerate(matrix):
      neighbors[i] = self.index.get_nns_by_vector(x, k)

    return neighbors

  def save(self):
    """
    """
    self.index.save(SAVE_PATH + self.get_id())

  def load_from_file(self, path: str):
    """
    """
    self.index.load(path)


