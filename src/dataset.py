"""Wrapper class for representing a dataset for this project."""
import numpy as np
import h5py
from sklearn.preprocessing import normalize
import faiss
import pickle


class BenchmarkDataset:
  """
  Description
  """
  def __init__(self, id: str, train: np.ndarray = None, test: np.ndarray = None, ground_truth: np.ndarray = None):
    self.id = id
    self.train = train
    self.test = test
    self.ground_truth = ground_truth

  def _normalized_train(self) -> np.ndarray:
    """
    Normalized dataset for methods.
    """
    return normalize(self.train, axis=1, norm="l2")

  def _normalized_test(self) -> np.ndarray:
    """
    Normalized dataset for methods.
    """
    return normalize(self.test, axis=1, norm="l2")

  def load_from_h5py(self, path: str):
    """
    """
    h5_data = h5py.File(path, "r")

    if not all(x in list(h5_data.keys()) for x in ["neighbors", "test", "train"]):
      raise Exception("Dataset does not seem to have required format.")

    self.train = np.zeros(h5_data["train"].shape, dtype="float32")
    h5_data["train"].read_direct(self.train)

    self.test = np.zeros(h5_data["test"].shape, dtype="float32")
    h5_data["test"].read_direct(self.test)

    # you might want to override this with compute_neighbors
    self.ground_truth = np.zeros(h5_data["neighbors"].shape, dtype=np.int32)
    h5_data["neighbors"].read_direct(self.ground_truth)

  def compute_neighbors(self, metric: str = "angular", k: int = 100):
    """
    Computes the nearest neighbors via exhaustive search and replaces self.ground_truth.
    The HDF5 datasets used here only include the top 100 nearest neighbors.
    """
    if metric != "angular":
      raise ValueError("only angular distance is supported.")

    validation_vector = self.ground_truth[0]
    
    index = faiss.IndexFlatL2(self.train.shape[1])
    index.add(self._normalized_train())

    _, self.ground_truth = index.search(self._normalized_test(), k)
    self.ground_truth = self.ground_truth.astype(np.int32)

    # double check the exhaustive search worked
    if not np.array_equal(self.ground_truth[0, 0:100], validation_vector[0:100]):
      raise Exception("Mismatch between computed neighbors and original ground truth.")

  def __str__(self):
    return "BenchmarkDataset(id={}, train={}, test={}, ground_truth={})".format(self.id, self.train.shape, self.test.shape, self.ground_truth.shape)
  

def get_dataset(path: str) -> BenchmarkDataset:
    """
    Loads a BenchmarkDataset object from a pickle file.
    """
    with open(path, "rb") as f:
        bd = pickle.load(f)
    return bd
