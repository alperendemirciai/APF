import numpy as np

from similarity.base import BaseSimilarity

class DTW(BaseSimilarity):
    """
    Class to compute Dynamic Time Warping (DTW) similarity between two time series.
    """

    def compute_similarity(self, idx1: np.ndarray, idx2: np.ndarray) -> float:
        """
        Compute the DTW similarity between two numpy arrays.
        :param idx1: First numpy array.
        :param idx2: Second numpy array.
        :return: DTW similarity between idx1 and idx2.
        """
        if not isinstance(idx1, np.ndarray) or not isinstance(idx2, np.ndarray):
            raise TypeError("Both inputs must be numpy arrays.")
        
        idx1 = idx1.flatten()
        idx2 = idx2.flatten()
        
        n, m = len(idx1), len(idx2)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0][0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(idx1[i - 1] - idx2[j - 1])
                dtw_matrix[i][j] = cost + min(dtw_matrix[i - 1][j],    # insertion
                                               dtw_matrix[i][j - 1],    # deletion
                                               dtw_matrix[i - 1][j - 1])  # match

        return -dtw_matrix[n][m]  # Negative because smaller distance means more similarity
    