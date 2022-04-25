"""This is a base class for all methods."""
import numpy as np
from config import BUILT_INDEX_DIR

class SearchIndex:
  """
  Base class for any search index class.
  """
  def __init__(self, name: str):
    self.name = name
    # intialize this attribute for child class
    self.index = None

  def is_initialized(self) -> bool:
    return self.index is None

  def get_id(self) -> str:
    return self.name + "_index"

