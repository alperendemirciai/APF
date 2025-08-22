from data.dataset import TimeSeriesDataset
from data.preprocessing import NumericalPreprocessing
from search.base import BaseSearcher


from similarity.base import BaseSimilarity
from similarity.cosine import CosineSimilarity
from similarity.dtw import DTW
from similarity.euclidean import EuclideanSimilarity
from similarity.manhattan import ManhattanSimilarity

from utils.visualization import plot_ohlc, plot_series
from utils.saver import get_unique_plot_dir, save_hyperparams
from utils.arguments import read_arguments, get_default_args, select_preprocessing_function, select_similarity_function

import numpy as np
import pandas as pd

import os
from typing import Dict, Any, List


def main(args:Dict[str, Any]=get_default_args()):

    print("Arguments:", args)

    preprocessing_function = select_preprocessing_function(args.preprocessing_function)
    
    # Create a dataset instance
    dataset = TimeSeriesDataset(data_path=args.data_path, preprocessing_function=preprocessing_function, fraction=args.fraction, columns=args.columns)
    
    if args.query_index is not None:
        IDX = args.query_index
        sequence = dataset.get(idx=IDX, sequence_length=args.sequence_length)
        sequence_ft = dataset.get(idx=IDX, sequence_length=args.sequence_length + args.future_length)
    elif args.query_csv is not None:
        query_df = pd.read_csv(args.query_csv, encoding='latin1', delimiter=',')
        if 'date' in query_df.columns:
            query_df.set_index('date', inplace=True)
        sequence = query_df[args.columns].values
        args.sequence_length = sequence.shape[0]
        sequence_ft = sequence.copy()
        IDX = 0

    
    print("First sequence:", sequence)

    sim_function = select_similarity_function(args.similarity_function)

    # Example usage of BaseSearcher
    searcher = BaseSearcher(search_space=dataset, similarity_function=sim_function)
    results = searcher.search(query=sequence, top_k=args.top_k, stride=args.stride, sequence_length=args.sequence_length, verbose=args.verbose)

    print(f"Top {args.top_k} similar sequences indices:", results)

    # --- automatic unique directory ---
    plot_dir = get_unique_plot_dir("plots")

    if args.plot_ohlc:
        plot_ohlc(sequence, save_path=os.path.join(plot_dir, "query_sequence.png"))
        plot_ohlc(sequence_ft, save_path=os.path.join(plot_dir, "query_sequence_future.png"))
    else:
        plot_series(sequence, save_path=os.path.join(plot_dir, "query_sequence.png"))
        plot_series(sequence_ft, save_path=os.path.join(plot_dir, "query_sequence_future.png"))

    future_results = []
    similarities = []
    for idx,sim in results:
        future_sequence = searcher.get_future(idx=idx, sequence_length=args.sequence_length, future_sequence_length=args.future_length)
        future_results.append(future_sequence)
        similarities.append(sim)
        if args.plot_ohlc:
            plot_ohlc(future_sequence, save_path=os.path.join(plot_dir, f"future_sequence_{idx}.png"))
        else:
            plot_series(future_sequence, save_path=os.path.join(plot_dir, f"future_sequence_{idx}.png"))

    # weighted_prediction = np.average(future_results, axis=0, weights=similarities)
    weighted_prediction = np.average(future_results, axis=0, weights=np.array(similarities) / np.sum(similarities))
    prediction = weighted_prediction

    if args.plot_ohlc:
        plot_ohlc(prediction, save_path=os.path.join(plot_dir, "predicted_future_sequence.png"))
    else:
        plot_series(prediction, save_path=os.path.join(plot_dir, "predicted_future_sequence.png"))
    print("Future sequences plotted and saved.")
    #print("Future sequences for top results:", future_results)
    print(f"All plots saved under: {plot_dir}")




if __name__ == "__main__":
    args = read_arguments()
    main(args)
