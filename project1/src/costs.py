import numpy as np


def mse(e):
    return 1/2 * np.mean(e**2)


def rmse(e):
    return np.sqrt(2 * mse(e))


def mae(e):
    return np.mean(np.abs(e))


def sigmoid(z):
    z = np.clip(z, -1000, 1000)
    return 1.0 / (1 + np.exp(-z))


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


def compute_log_likelihood_error(y, tx, w, lambda_=0.0):
    """Compute the loss of the log-likelihood cost function."""
    tx_dot_w = tx.dot(w)
    return np.sum(np.log(1 + np.exp(tx_dot_w))) - y.dot(tx_dot_w) + 2.0 * lambda_ * w.dot(w)
