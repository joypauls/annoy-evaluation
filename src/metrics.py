"""Metrics used in project."""

def average_recall(ground_truth, estimated, k=5):
    """Fetches rows from a Smalltable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle.  String keys will be UTF-8 encoded.

    Args:
        ground_truth: An open smalltable.Table instance.
        estimated: A sequence of strings representing the key of each table
          row to fetch.  String keys will be UTF-8 encoded.
        k: If True only rows with values set for all keys will be
          returned.

    Returns:
        A score
    """
    if len(estimated) > k:
        estimated = estimated[:k]

    if ground_truth is None:
        return 0.0

    score = 0.0
    num_hits = 0.0  # keep as float to avoid type issue in division
    for i, p in enumerate(estimated):
        if p in ground_truth and p not in estimated[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(ground_truth), k)