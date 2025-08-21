import numpy as np
from typing import List, Any
from base import BaseSearcher
import pandas as pd


class FFTSearcher(BaseSearcher):
    """
    Class to perform Fast Fourier Transform (FFT) based search on time series data.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize the FFTSearcher with a DataFrame containing time series data.
        :param data: DataFrame containing time series data.
        """
        super().__init__(data)
    
    def search(self, query: np.ndarray, threshold: float = 0.5, top_k:int =10) -> List[int]:
        """
        Search for the top_k most similar items to the query using FFT.
        :param query: A numpy array representing the query item.
        :param threshold: Minimum similarity threshold to consider an item as similar.
        :param top_k: The number of top similar items to return.
        :return: A list of indices of the top_k most similar items.
        """
        if not isinstance(query, np.ndarray):
            raise TypeError("query must be a numpy array.")
        
        fft_query = np.fft.fft(query)
        similarities = []
        
        for idx in range(len(self.search_space)):
            item = self.search_space.get(idx)
            fft_item = np.fft.fft(item)
            similarity = np.abs(np.dot(fft_query, fft_item)) / (np.linalg.norm(fft_query) * np.linalg.norm(fft_item))
            
            if similarity >= threshold:
                similarities.append((idx, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in similarities[:top_k]]