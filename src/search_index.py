"""This is a base class for all methods."""

from config import BUILT_INDEX_DIR

class SearchIndex:
  def __init__(self, training_data, method_name):
    self.training_data = training_data
    self.method_name = method_name
    # intialize this attribute
    self.index = None

