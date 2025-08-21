import numpy as np


class NumericalPreprocessing:
    """
    A class containing various numerical preprocessing functions.
    """

    @staticmethod
    def minmax_norm(x: np.array) -> np.ndarray:
        """
        Normalize a numpy array using Min-Max normalization.
        :param x: Input numpy array.
        :return: Normalized numpy array.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        
        min_val = np.min(x, axis=0) + np.finfo(float).eps  # Avoid division by zero
        max_val = np.max(x, axis=0) + np.finfo(float).eps  # Avoid division by zero
        
        
        return (x - min_val) / (max_val - min_val)
    
    @staticmethod
    def z_score_norm(x: np.array) -> np.ndarray:
        """
        Normalize a numpy array using Z-score normalization.
        :param x: Input numpy array.
        :return: Normalized numpy array.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        
        mean = np.mean(x)
        std = np.std(x)
        
        if std == 0:
            return np.zeros_like(x)  # Avoid division by zero
        
        return (x - mean) / std
