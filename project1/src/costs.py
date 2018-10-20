import numpy as np


def mse(e):
    return 1/2 * np.mean(e**2)


def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    return mse(e)