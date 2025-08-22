import numpy as np
from typing import List, Any
from search.base import BaseSearcher
import pandas as pd


class FFTSearcher(BaseSearcher):
    """
    Class to perform Fast Fourier Transform (FFT) based search on time series data.
    """

    def __init__(self, search_space: pd.DataFrame):
        """
        Initialize the FFTSearcher with a DataFrame containing time series data.
        :param data: DataFrame containing time series data.
        """
        super().__init__(search_space=search_space)
    
    def search(self, query: np.ndarray, threshold: float = 0.5, top_k:int =10, sequence_length:int=30, stride:int=1, verbose:bool = True) -> List[int]:
        """
        Search for the top_k most similar items to the query using FFT.
        :param query: A numpy array representing the query item.
        :param threshold: Minimum similarity threshold to consider an item as similar.
        :param top_k: The number of top similar items to return.
        :return: A list of indices of the top_k most similar items.
        """
        if not isinstance(query, np.ndarray):
            raise TypeError("query must be a numpy array.")
        
        query = query.flatten()
        fft_query = np.fft.fft(query)
        similarities = []
        
        for idx in range(0, len(self.search_space), stride):
            #print(f"Processing index: {idx}")
            if idx % 50 == 0 and verbose:
                print(f"Processing index: {idx} of {len(self.search_space)}")
            if idx + sequence_length > len(self.search_space):
                break
            
            item = self.search_space.get(idx, sequence_length)

            item = item.flatten()

            fft_item = np.fft.fft(item)
            similarity = np.abs(np.dot(fft_query, fft_item)) / (np.linalg.norm(fft_query) * np.linalg.norm(fft_item))
            
            if similarity >= threshold:
                similarities.append((idx, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [(idx, _) for idx, _ in similarities[:top_k]]