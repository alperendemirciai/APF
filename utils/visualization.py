import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List

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




def plot_series(series: np.ndarray, save_path: str = "series_plot.png"):
    """
    Plot a time series and save it.

    Parameters
    ----------
    series : np.ndarray
        A 1D numpy array representing the time series.
    save_path : str
        File path to save the plot (e.g., 'plots/series_plot.png').
    """
    if not isinstance(series, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    
    plt.figure(figsize=(12, 6))
    plt.plot(series, label='Time Series', color='blue')
    plt.title('Time Series Plot')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    
    # Ensure directory exists automatically
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved time series plot to {save_path}")
