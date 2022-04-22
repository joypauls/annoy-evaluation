# Benchmarking script
from annoy import AnnoyIndex
import numpy as np

dim = 36  # Length of item vector that will be indexed

np.random.seed(1312)

t = AnnoyIndex(dim, "angular")
for i in range(1000):
    v = np.random.normal(0, 1, dim)
    t.add_item(i, v)

t.build(10)  # 10 trees
t.save("ml100k.annoy")

# ...

u = AnnoyIndex(dim, "angular")
u.load("ml100k.annoy")  # super fast, will just mmap the file
print(u.get_nns_by_item(0, 10))  # will find the 10 nearest neighbors
