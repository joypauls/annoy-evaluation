"""Wrapper class for representing a dataset for this project."""
import numpy as np

class Dataset:
  def __init__(self, train: np.ndarray, test: np.ndarray, ground_truth: np.ndarray):
    self.train = train
    self.test = test
    self.ground_truth = ground_truth



