import numpy as np
from data.dataset import TimeSeriesDataset
from similarity.base import BaseSimilarity
from typing import List, Any

class BaseSearcher:
    """
    A base class for search algorithms.
    This class should not be instantiated directly.
    """

    def __init__(self, search_space: TimeSeriesDataset, similarity_function:BaseSimilarity=None):
        """
        Initialize the searcher with a search space and a similarity function.
        :param search_space: An instance of TimeSeriesDataset representing the search space.
        :param similarity_function: A function to compute similarity between items in the search space.
        """
        if not isinstance(search_space, TimeSeriesDataset):
            raise TypeError("search_space must be an instance of TimeSeriesDataset.")
        if similarity_function is not None and not isinstance(similarity_function, BaseSimilarity):
            raise TypeError("similarity_function must be an instance of BaseSimilarity or None.")
        
        self.search_space = search_space
        self.similarity_function = similarity_function 

    def search(self, query: np.ndarray, top_k: int = 10, stride:int = 1, sequence_length:int = 30, verbose:bool=True) -> List[int]:
        """
        Search for the top_k most similar items to the query in the search space.
        :param query: A numpy array representing the query item.
        :param top_k: The number of top similar items to return.
        :return: A list of indices of the top_k most similar items.
        """
        if not isinstance(query, np.ndarray):
            raise TypeError("query must be a numpy array.")
        
        similarities = []
        print(f"Self.search_space: {len(self.search_space)}")
        for idx in range(0, len(self.search_space), stride):
            #print(f"Processing index: {idx}")
            if idx % 50 == 0 and verbose:
                print(f"Processing index: {idx} of {len(self.search_space)}")
            if idx + sequence_length > len(self.search_space):
                break
            
            item = self.search_space.get(idx, sequence_length)
            similarity = self.similarity_function.compute_similarity(query, item)
            similarities.append((idx, similarity))

        #print(similarities)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [(idx,_) for idx, _ in similarities[:top_k]]
    
    def get_future(self, idx: int, sequence_length:int = 30, future_sequence_length: int = 30) -> np.ndarray:
        """
        Get the future sequence from the dataset.
        :param idx: Index to start from.
        :param sequence_length: Length of the sequence to retrieve.
        :return: Numpy array of the future sequence.
        """
        if idx < 0 or idx >= len(self.search_space):
            raise IndexError("Index out of bounds.")
        
        if idx + sequence_length + future_sequence_length > len(self.search_space):
            return np.array([])
            raise IndexError("Index and future sequence length exceed dataset length.")
        
        return self.search_space.get(idx, sequence_length + future_sequence_length)
    
    def get_prediction(self, idx: int, sequence_length:int = 30, future_sequence_length: int = 30) -> np.ndarray:
        """
        Get the prediction sequence from the dataset.
        :param idx: Index to start from.
        :param sequence_length: Length of the sequence to retrieve.
        :return: Numpy array of the prediction sequence.
        """
        future = self.get_future(idx, sequence_length, future_sequence_length)
        if future.size == 0:
            return np.array([])
        
        future = future[-future_sequence_length:]
        future = np.mean(future, axis=1)  # Average the future sequence
        return future