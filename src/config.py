# where a built index is saved
BUILT_INDEX_DIR = "./models/"

# paths for datasets
RAW_PATHS = [
    "./data/raw/glove-25-angular.hdf5",
    "./data/raw/glove-50-angular.hdf5",
    "./data/raw/glove-100-angular.hdf5",
]
PROCESSED_PATHS = [
    "./data/glove_25_angular.pkl",
    "./data/glove_50_angular.pkl",
    "./data/glove_100_angular.pkl",
]
DATASET_IDS = ["glove25", "glove50", "glove100"]

# number of neighbors for evaluation
K_NEIGHBORS_OVERALL = 100
K_NEIGHBORS_MAX = 300

# hyperparameters for methods
# TODO: these should really be tuned and studied as well
N_TREES = 20
N_LIST = 20
SEARCH_K = 2000
