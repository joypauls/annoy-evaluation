# Benchmarking script
from annoy import AnnoyIndex
import numpy as np
import pickle

dim = 36  # Length of item vector that will be indexed
n = 1000
np.random.seed(1312)

random_data =  np.array([np.random.normal(0, 1, dim) for i in range(0, n)])

def build_bf_index():
  with open("./data/random.bf", "wb") as f:
    pickle.dump(random_data, f)

def build_annoy_index():
  t = AnnoyIndex(dim, "angular")
  for i, x in enumerate(random_data):
    t.add_item(i, x)
  t.build(10) # 10 trees
  t.save("./data/random.annoy")

build_annoy_index()
build_bf_index()

u = AnnoyIndex(dim, "angular")
u.load("./data/random.annoy") # super fast, will just mmap the file
print(u.get_nns_by_item(0, 10)) # will find the 10 nearest neighbors
