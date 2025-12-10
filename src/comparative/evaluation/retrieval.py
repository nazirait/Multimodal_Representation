# src/comparative/evaluation/retrieval.py
import numpy as np

def recall_at_k(scores: np.ndarray, ground_truth: np.ndarray, k: int = 1):
    """
    Compute recall@k for retrieval tasks.
    scores: [N, N] similarity matrix (higher is better).
    ground_truth: [N] correct index for each query (often np.arange(N)).
    """
    N = scores.shape[0]
    ranks = np.argsort(-scores, axis=1)   # descending
    hits = np.any([ground_truth[i] in ranks[i, :k] for i in range(N)])
    recall = np.mean([ground_truth[i] in ranks[i, :k] for i in range(N)])
    return recall

def mean_reciprocal_rank(scores: np.ndarray, ground_truth: np.ndarray):
    """
    Compute mean reciprocal rank (MRR).
    """
    N = scores.shape[0]
    ranks = np.argsort(-scores, axis=1)
    mrr = 0.0
    for i in range(N):
        rank = np.where(ranks[i] == ground_truth[i])[0][0] + 1  # 1-based
        mrr += 1.0 / rank
    return mrr / N
