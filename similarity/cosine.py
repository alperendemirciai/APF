import numpy as np

from similarity.base import BaseSimilarity

class CosineSimilarity(BaseSimilarity):
    """
    Class to compute cosine similarity between two vectors.
    """

    def compute_similarity(self, idx1: np.ndarray, idx2: np.ndarray) -> float:
        """
        Compute the cosine similarity between two numpy arrays.
        :param idx1: First numpy array.
        :param idx2: Second numpy array.
        :return: Cosine similarity between idx1 and idx2.
        """
        if not isinstance(idx1, np.ndarray) or not isinstance(idx2, np.ndarray):
            raise TypeError("Both inputs must be numpy arrays.")
        
        #print(f"Computing cosine similarity between arrays of shapes {idx1.shape} and {idx2.shape}")
        
        if idx1.shape != idx2.shape:
            raise ValueError("Input arrays must have the same shape.")
        
        norm1 = np.linalg.norm(idx1)
        norm2 = np.linalg.norm(idx2)

        idx1 = idx1.flatten()
        idx2 = idx2.flatten()
        
        if norm1 == 0 or norm2 == 0:
            return 0.0  # Avoid division by zero
        
        return np.dot(idx1, idx2) / (norm1 * norm2)