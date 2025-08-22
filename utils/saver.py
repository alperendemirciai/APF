import os
import uuid
from datetime import datetime


def get_unique_plot_dir(base: str = "plots") -> str:
    """
    Create a unique directory for plots inside the given base directory.
    Uses timestamp + random uuid to avoid collisions.
    """
    unique_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    plot_dir = os.path.join(base, unique_id)
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir

def save_hyperparams(hyperparams: dict, save_path: str):
    """
    Save hyperparameters to a file.
    
    Parameters
    ----------
    hyperparams : dict
        Dictionary containing hyperparameters.
    save_path : str
        File path to save the hyperparameters (e.g., 'hyperparams.json').
    """
    import json
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(hyperparams, f, indent=4)
    
    print(f"Hyperparameters saved to {save_path}")
