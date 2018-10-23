"""Implementations."""

import numpy as np

import costs
import gradients

from util import parsers

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression usign Gradient Descent."""
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = gradients.mse_gradient(y, tx, w)
        loss = costs.mse(err)
        # gradient w by descent update
        w = w - gamma * grad
        print("GD({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters-1, l=loss))
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression usign Stochastic Gradient Descent."""
    w = initial_w
    for n_iter in range(max_iters):
        for y_batch, tx_batch in parsers.batch_iter(
                y, tx, batch_size=1, num_batches=1):
            # compute a stochastic gradient and loss
            grad, err = gradients.mse_gradient(y_batch, tx_batch, w)
            loss = costs.mse(err)
            # update w through the stochastic gradient update
            w = w - gamma * grad
        print("SGD({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters-1, l=loss))
    return w, loss


def least_squares(y, tx):
    """Least squares using normal equations."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)

    w = np.linalg.solve(a, b)
    loss = costs.compute_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""
    N, D = tx.shape
    lambda_ = 2 * N * lambda_

    a = tx.T.dot(tx) + lambda_ * np.eye(D)
    b = tx.T.dot(y)

    w = np.linalg.solve(a, b)
    loss = costs.compute_rmse(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using GD or SGD."""
    w = initial_w

    for n_iter in range(max_iters):
        delta = np.linalg.inv(gradients.hessian(w, tx))
        w = w - gamma * delta.dot(gradients.log_likelihood_gradient(y, tx, w))
        loss = costs.compute_log_likelihood_error(y, tx, w)
        print("LOG({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters-1, l=loss))
    
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularised logistic regression using GD or SGD."""
    pass