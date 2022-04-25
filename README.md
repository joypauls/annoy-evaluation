# Annoy Similarity Search Evaluation

> :warning: This works, but is a WIP and under development.

This is a proof-of-concept evaluation of a popular Python (C++ under the hood!) library for Approximate Nearest Neighbors (ANN) similarity search. Its performance, as measured by [recall](https://en.wikipedia.org/wiki/Precision_and_recall#Recall), is measured for a few different datasets (see [below](#datasets)):
- [Annoy (Spotify)](https://github.com/spotify/annoy)
  - Builds search index using a tree-based algorithm

The library below is alos utilized to perform a fast and efficient *exhaustive search* to produce ground truth nearest neighbors to measure recall against.
- [Faiss (Facebook)](https://github.com/facebookresearch/faiss)
  - Uses hash table with Locality Sensitive Hashing (LSH) as a search index

Notes for future improvements:
- Time permitting, I will also be adding other methods to the evaluation phase, such as Faiss
- Note that this study does *not* measure lookup speeds (ie queries/second) which is a major component of choosing a library to use for this type of task. The the [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) project has in depth information on that.
- Should really be run inside a docker container for better reproducibility and packaging of python environment.
- Once API is stable after implementing a few methods, packaging `src` could be nice.

## Datasets

The datasets used here are from the Global Vectors for Word Representation (GloVe) project ([link to original paper](https://nlp.stanford.edu/pubs/glove.pdf)).

- Info about the GloVe datasets can be found [here](https://nlp.stanford.edu/projects/glove/). In short, they contain embeddings of words and are commonly used to benchmark similarity search tasks.
- The specific files used in this project are retrieved from the [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) project and transformed further:
  - glove25: GloVe with 25-dimensional embedding vectors
    - training set size: 1,183,514
    - test set size: 10,000
  - glove50: GloVe with 50-dimensional embedding vectors
    - training set size: 1,183,514
    - test set size: 10,000
  - glove100: GloVe with 100-dimensional embedding vectors
    - training set size: 1,183,514
    - test set size: 10,000


## Results

The plot below shows the distribution of average recall@100 (that is, recall when retrieving the 100 nearest neighbors) for each query vector in the test set of each dataset. The vertical line of the same color shows the Mean Average Recall (MAR@100) for the entire test set.

![preliminary results](./plots/recall_distribution.png)

## Methods

### Evaluation Metric

Average recall@k differs from recall@k by being *rank-aware*. For recall@k, it is calculated by:

$$
\text{Recall@k} = \frac{\text{\# of relevant matches retrieved out of k predictions}}{\text{k}}
$$

As long as the set of predictions is identical, recall@k doesn't care about ordering. Average recall attempts to capture some of that with the following algorithm sketched out below:

```
Given an unordered list of top k true matches (V)
Given an ordered list of top k estimated matches (E)
Initialize empty list of recall scores R
For each true positive match TP in E:
  Calculate recall@i, where i is the index of TP in E
  Append this result to R
Return mean(R)
```

This algorithm produces scores that are higher if more true positives are ordered higher in the list of predictions, making it more suitable for tasks that care about ranking.


### Distance Function

For this project, I have used [angular distance](https://en.wikipedia.org/wiki/Cosine_similarity#Angular_distance_and_similarity) as the distance metric for definining what closeness between vectors in our vector spaces means. Angular distance is closely related to *cosine similairty*, except unlike cosine similarity, it is a metric in the mathematical sense and makes the set of training vectors + our distance metric an actual metric space.

Under the hood, Annoy actually doesn't directly calculate angular distances, it calculates the Euclidean distance between $L_2$-normalized vectors. Why does this work? Because both angular distance and Euclidean distance between $L_2$-normalized vectors are monotonic transformations of cosine similarity, meaning that all three produce the same ordering!

---

## Run It Yourself!

Tested Python version: 3.9.1

Make sure to set up a python virtual environment of your choosing (I used miniconda) and install the dependencies in `requirements.txt`.

Note: if you have multiple python installations you might want to use `python3` instead of `python`. 

### 1. Download Raw Datasets

```
sh download.sh
```

Time: a couple minutes, depends on network.

### 2. Preprocess Datasets

Transform the datasets into a cleaner format.

```
python preprocess.py
```

Time: 2-3 minutes

### 3. Build the Indexes

Build the Annoy search indexes for each dataset.

```
python build_indexes.py
```

Time: ~1 minutes

### 4. Generate Results

Time: <1 minute

```
python generate_results.py
```

### 5. Evaluate

Time: <1 minute

```
python evaluate.py
```



