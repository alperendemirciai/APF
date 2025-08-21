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
    
    @staticmethod
    def max_norm(x: np.ndarray) -> np.ndarray:
        """
        Normalize a numpy array using Max normalization.
        :param x: Input numpy array.
        :return: Normalized numpy array.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        
        max_val = np.max(x, axis=0) + np.finfo(float).eps  # Avoid division by zero
        return x / max_val
    
    @staticmethod
    def backward_roc(x: np.ndarray, n:int = 1) -> np.ndarray:
        """
        Calculate the backward rate of change (ROC) of a numpy array.
        :param x: Input numpy array.
        :param n: Number of periods to look back.
        :return: Backward ROC as a numpy array.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a numpy array.")
        
        if n < 1:
            raise ValueError("n must be at least 1.")
        
        roc = np.zeros_like(x)
        roc[n:] = (x[n:] - x[:-n]) / x[:-n]
        
        return roc