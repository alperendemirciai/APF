import argparse
import os
import sys
from typing import List, Any, Dict
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_default_args() -> Dict[str, Any]:
    """
    Get default arguments for the script.
    :return: A dictionary containing default arguments.
    """
    return {
        'data_path': 'datasets/1_minute/btc_data_2017min.csv',
        'sequence_length': 0,
        'future_length': 10,
        'top_k': 15,
        'stride': 30,
        'fraction': 1.00,
        'columns': ['open', 'high', 'low', 'close'],
        'preprocessing_function': 'max_norm',
        'similarity_function': 'euclidean',
        'output_dir': 'plots',
        'query_index': 0,
        'query_csv': None,
        'verbose': False,
        'save_hyperparams': False,
        'save_plots': False
    }


def read_arguments() -> argparse.Namespace:
    """
    Read command line arguments and return them as a Namespace object.
    """
    parser = argparse.ArgumentParser(description="Run Brute Force Search on Time Series Data")
    parser.add_argument('--data_path', type=str, default='datasets/1_minute/btc_data_2017min.csv', help='Path to the dataset CSV file', required=True)
    parser.add_argument('--sequence_length', type=int, default=0, help='Length of the sequence to retrieve')
    parser.add_argument('--future_length', type=int, default=10, help='Length of the future sequence to retrieve')
    parser.add_argument('--top_k', type=int, default=15, help='Number of top similar sequences to retrieve', required=True)
    parser.add_argument('--stride', type=int, default=30, help='Stride for searching similar sequences')
    parser.add_argument('--fraction', type=float, default=1.00, help='Fraction of the dataset to use')
    parser.add_argument('--columns', type=str, nargs='+', default=['open', 'high', 'low', 'close'], help='Columns to use from the dataset')
    parser.add_argument('--preprocessing_function', type=str, default='max_norm', help='Preprocessing function to apply on the dataset', required=True)
    parser.add_argument('--similarity_function', type=str, default='euclidean', help='Similarity function to use for searching', required=True)
    parser.add_argument('--output_dir', type=str, default='plots', help='Directory to save the output plots')
    parser.add_argument('--query_index', type=int, default=None, help='Integer index of the query sequence in the dataset. Use this if you want to use a specific index instead of a CSV file.')
    parser.add_argument('--query_csv', type=str, default=None, help='Path to a CSV file containing the query sequence if not using index')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--save_hyperparams', action='store_true', help='Save hyperparameters to a file')
    parser.add_argument('--save_plots', action='store_true', help='Save plots of the query and future sequences')

    return parser.parse_args()

def select_preprocessing_function(name: str):
    """
    Select the preprocessing function based on the name.
    :param name: Name of the preprocessing function.
    :return: The corresponding preprocessing function.
    """
    from data.preprocessing import NumericalPreprocessing
    if name == 'max_norm':
        return NumericalPreprocessing.max_norm
    elif name == 'minmax_norm':
        return NumericalPreprocessing.minmax_norm
    elif name == 'z_score_norm':
        return NumericalPreprocessing.z_score_norm
    elif name == 'backward_roc':
        return NumericalPreprocessing.backward_roc
    else:
        raise ValueError(f"Unknown preprocessing function: {name}")
    
def select_similarity_function(name: str):
    """
    Select the similarity function based on the name.
    :param name: Name of the similarity function.
    :return: The corresponding similarity function class.
    """
    
    if name == 'euclidean':
        from similarity.euclidean import EuclideanSimilarity
        return EuclideanSimilarity()
    elif name == 'manhattan':
        from similarity.manhattan import ManhattanSimilarity
        return ManhattanSimilarity()
    elif name == 'cosine':
        from similarity.cosine import CosineSimilarity
        return CosineSimilarity()
    elif name == 'dtw':
        from similarity.dtw import DTW
        return DTW()
    elif name == 'correlation':
        from similarity.correlation import CorrelationSimilarity
        return CorrelationSimilarity()
    else:
        raise ValueError(f"Unknown similarity function: {name}")