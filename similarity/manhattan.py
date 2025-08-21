import numpy as np

from similarity.base import BaseSimilarity

class ManhattanSimilarity(BaseSimilarity):
    """
    Class to compute Manhattan similarity between two vectors.
    """

    def compute_similarity(self, idx1: np.ndarray, idx2: np.ndarray) -> float:
        """
        Compute the Manhattan similarity between two numpy arrays.
        :param idx1: First numpy array.
        :param idx2: Second numpy array.
        :return: Manhattan similarity between idx1 and idx2.
        """
        if not isinstance(idx1, np.ndarray) or not isinstance(idx2, np.ndarray):
            raise TypeError("Both inputs must be numpy arrays.")
        
        if idx1.shape != idx2.shape:
            raise ValueError("Input arrays must have the same shape.")
        
        return -np.sum(np.abs(idx1 - idx2))  # Negative because smaller distance means more similarity