import numpy as np

from similarity.base import BaseSimilarity

class CorrelationSimilarity(BaseSimilarity):
    """
    Class to compute correlation similarity between two vectors.
    """

    def compute_similarity(self, idx1: np.ndarray, idx2: np.ndarray) -> float:
        """
        Compute the correlation similarity between two numpy arrays.
        :param idx1: First numpy array.
        :param idx2: Second numpy array.
        :return: Correlation similarity between idx1 and idx2.
        """
        if not isinstance(idx1, np.ndarray) or not isinstance(idx2, np.ndarray):
            raise TypeError("Both inputs must be numpy arrays.")
        
        if idx1.shape != idx2.shape:
            raise ValueError("Input arrays must have the same shape.")
        
        correlation_matrix = np.corrcoef(idx1, idx2)
        return correlation_matrix[0, 1]  # Return the correlation coefficient
    
    