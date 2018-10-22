import numpy as np


def mse(e):
    return 1/2 * np.mean(e**2)


def rmse(e):
    return np.sqrt(2 * mse(e))


def mae(e):
    return np.mean(np.abs(e))


def compute_mse(y, tx, w):
    """Compute the loss by MSE (Mean Square Error)."""
    e = y - tx.dot(w)
    return mse(e)


def compute_rmse(y, tx, w):
    """Compute the loss by RMSE (Root Mean Square Error)."""
    e = y - tx.dot(w)
    return rmse(e)


def compute_mae(y, tx, w):
    """Compute the loss by MAE (Mean Absolute Error)."""
    e = y - tx.dot(w)
    return mae(e)