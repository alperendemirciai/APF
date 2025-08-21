import numpy as np
import pandas as pd
import os
import sys
from typing import List, Any
from data.preprocessing import NumericalPreprocessing


class TimeSeriesDataset:
    def __init__(self, data_path:str=None, preprocessing_function:NumericalPreprocessing=None, fraction:float=1.0) -> None:
        assert data_path.endswith('.csv'), "Data path must be a CSV file."
        assert os.path.exists(data_path), "Data path does not exist."

        self.data_path = data_path
        self.data = self.__load_data(data_path, fraction)
        self.fraction = fraction
        self.preprocessing_function = preprocessing_function if preprocessing_function else lambda x: x

    def __load_data(self, data_path:str=None, fraction:float=1.0) -> pd.DataFrame:
        """
        Load the dataset from the specified path.
        :param data_path: Path to the dataset file.
        :return: Loaded dataset as a pandas DataFrame.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file {data_path} does not exist.")
        
        data = pd.read_csv(data_path, encoding='latin1', delimiter=',')
        print(f"Data loaded from {data_path}. Shape: {data.shape}")
        if 'date' in data.columns:
            data.set_index('date', inplace=True)
        if fraction < 1.0:
            data = data.iloc[:int(data.shape[0] * fraction)]
        #cols = data.columns.tolist()
        #data.drop([col for col in cols if col != 'close'], inplace=True, axis=1)
        data.drop("volume", axis=1, inplace=True, errors='ignore')
        print(f"Data shape after loading and preprocessing: {data.shape}")
        return data


    def __len__(self) -> int:
        #print(f"Dataset length: {self.data.shape[0]}")
        return self.data.shape[0]
    
    def crop_last(self, sequence_length:int=30) -> None:
        """
        Crop the last sequence of the dataset to the specified length.
        :param sequence_length: Length of the sequence to crop.
        """
        if len(self.data) < sequence_length:
            raise ValueError("Dataset is smaller than the specified sequence length.")
        
        self.data = self.data.iloc[-sequence_length:]

    def get(self, idx, sequence_length:int= 30) -> np.ndarray:
        if idx < 0 or idx >= len(self.data):
            raise IndexError("Index out of bounds.")
    
        if idx + sequence_length > len(self.data):
            raise IndexError("Index and sequence length exceed dataset length.")
        
        return self.preprocessing_function(np.array(self.data.iloc[idx : idx + sequence_length ].values))
    

    