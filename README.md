# Approximate NN Method Evaluation

This is a comparison of two popular Python (C++ under the hood!) libraries for Approximate Nearest Neighbors (ANN) that take different approaches:
- [Annoy (Spotify)](https://github.com/spotify/annoy)
  - Builds search index using a tree-based algorithm
- [Faiss (Facebook)](https://github.com/facebookresearch/faiss)
  - Uses hash table with Locality Sensitive Hashing (LSH) as a search index

## Results


## Methods


## About The Data

---

## Run It Yourself!

### 1. Download Datasets

Time: a couple minutes, depends on network.

```
sh download.sh
```

### 2. Preprocess Datasets

Time: 2-3 minutes

```
python preprocess.py
```

### 3. Build the Indexes

Time: 2-3 minutes

```
python build_indexes.py
```

### 4. Generate Results

Time: 2-3 minutes

```
python generate_results.py
```

### 5. Evaluate

Time: 2-3 minutes

```
python evaluate.py
```


## Code Structure

```
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│── models
│   ├── random.bf       <- Data from third party sources.
│   ├── random.annoy        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── notebooks          <- Jupyter notebooks
│
├── requirements.txt   <- The pip requirements file
│
├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
```




