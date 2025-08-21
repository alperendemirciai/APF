from data.dataset import TimeSeriesDataset
from data.preprocessing import NumericalPreprocessing
from search.base import BaseSearcher
from similarity.base import BaseSimilarity
from similarity.cosine import CosineSimilarity
from similarity.dtw import DTW
from similarity.euclidean import EuclideanSimilarity
from similarity.manhattan import ManhattanSimilarity

import numpy as np
import pandas as pd

import os
import uuid
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def get_unique_plot_dir(base: str = "plots") -> str:
    """
    Create a unique directory for plots inside the given base directory.
    Uses timestamp + random uuid to avoid collisions.
    """
    unique_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    plot_dir = os.path.join(base, unique_id)
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


def plot_ohlc(ohlc: np.ndarray, save_path: str = "ohlc_plot.png"):
    """
    Plot OHLC candlestick chart from an N x 4 numpy array and save it.

    Parameters
    ----------
    ohlc : np.ndarray
        An N x 4 numpy array where columns are [Open, High, Low, Close].
    save_path : str
        File path to save the chart (e.g., 'plots/future_sequence.png').
    """
    if not isinstance(ohlc, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if ohlc.shape[1] != 4:
        raise ValueError("Input must be an N x 4 numpy array (Open, High, Low, Close).")

    # --- ensure directory exists automatically ---
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    opens = ohlc[:, 0]
    highs = ohlc[:, 1]
    lows = ohlc[:, 2]
    closes = ohlc[:, 3]
    n = len(ohlc)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(n):
        # Wick (high-low line)
        ax.plot([x[i], x[i]], [lows[i], highs[i]], color="black", linewidth=1)

        # Candle body
        color = "green" if closes[i] >= opens[i] else "red"
        lower = min(opens[i], closes[i])
        height = abs(closes[i] - opens[i])
        rect = Rectangle((x[i] - 0.3, lower), 0.6, height, color=color, alpha=0.8)
        ax.add_patch(rect)

    ax.set_xlim(-1, n)
    ax.set_title("OHLC Candlestick Chart")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved candlestick chart to {save_path}")


def main():
    # Example usage of TimeSeriesDataset
    data_path = 'datasets/1_hour/BTC-1Hour.csv'
    preprocessing_function = NumericalPreprocessing.minmax_norm
    
    # Create a dataset instance
    dataset = TimeSeriesDataset(data_path=data_path, preprocessing_function=preprocessing_function, fraction=0.05)
    
    # Get a specific sequence from the dataset
    sequence = dataset.get(idx=100, sequence_length=50)
    print("First sequence:", sequence)

    # Choose similarity function
    # sim_function = DTW()
    # sim_function = CosineSimilarity()
    # sim_function = ManhattanSimilarity()
    # sim_function = EuclideanSimilarity()
    sim_function = CosineSimilarity()

    # Example usage of BaseSearcher
    searcher = BaseSearcher(search_space=dataset, similarity_function=sim_function)
    results = searcher.search(query=sequence, top_k=30, stride=10, sequence_length=50)
    print("Top 5 similar sequences indices:", results)

    # --- automatic unique directory ---
    plot_dir = get_unique_plot_dir("plots")

    future_results = []
    for idx in results:
        idx = idx[0]
        future_sequence = searcher.get_future(idx=idx, sequence_length=50, future_sequence_length=10)
        future_results.append(future_sequence)
        plot_ohlc(future_sequence, save_path=os.path.join(plot_dir, f"future_sequence_{idx}.png"))

    #print("Future sequences for top results:", future_results)
    print(f"All plots saved under: {plot_dir}")


if __name__ == "__main__":
    main()
