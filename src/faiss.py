"""Faiss"""
import numpy as np
import faiss

from src.search_index import SearchIndex
from src.config import N_LIST


class Faiss(SearchIndex):
    """
    Description
    """

    def __init__(self, dim: int, name: str = "faiss"):
        """
        Ideally the name passed includes context for the training data.
        """
        super().__init__(name=name)
        # initialize index
        quantizer = faiss.IndexFlatL2(dim)
        self.index = faiss.IndexIVFFlat(quantizer, dim, N_LIST, faiss.METRIC_L2)

    def build(self, training_data: np.ndarray, n_list: int = 10):
        """
        Build the index.
        """
        if len(training_data.shape) != 2:
            raise ValueError("training_data has the wrong shape.")

        try:
            self.index.train(training_data)
        except Exception as e:
            raise e

    def query(self, vector: np.ndarray, k: int) -> np.ndarray:
        """
        Returns k nearest neighbors by id.
        """
        if len(vector.shape) == 1:
            vector = vector.reshape((1, vector.shape[0]))

        _, neighbors = self.index.search(vector, k)
        return np.array(neighbors, dtype=np.int32)

    def batch_query(self, matrix: np.ndarray, k: int) -> np.ndarray:
        """
        Return nearest neighbors for each row vector.
        """
        # faiss CAN batch!
        _, neighbors = np.array(self.index.search(matrix, k), dtype=np.int32)
        return neighbors

    def save(self, dir: str):
        """
        Save built index.
        """
        faiss.write_index(self.index, dir + self.get_id())

    def load_from_file(self, path: str):
        """
        Load built index.
        """
        self.index = faiss.read_index(path)
