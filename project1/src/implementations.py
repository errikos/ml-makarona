"""Implementations."""

import numpy as np

import costs
import gradients


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = gradients.compute_gradient(y, tx, w)
        loss = costs.mse(err)
        # gradient w by descent update
        w = w - gamma * grad
        print("GD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return w, loss


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