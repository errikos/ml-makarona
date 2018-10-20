"""Implementations."""

import numpy as np
import costs


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    pass


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    pass


def least_squares(y, tx):
    """calculate the least squares."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)

    w = np.linalg.solve(a, b)
    loss = costs.compute_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    pass


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    pass


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    pass