import numpy as np


def separate_labels(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Load data from a csv file. Add intercept to the data.

    Args:
        filename (str): Path to the csv file.
        dtype (type): Type of the data.

    Returns:
        data (np.ndarray): tuple of (X, y) where X is the data with intercept and y are the labels.
    """
    X = data[:, :-1]
    y = data[:, -1]
    X = np.c_[np.ones(X.shape[0]), X]
    return X, y
