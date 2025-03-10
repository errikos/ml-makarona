"""Implementations."""

import numpy as np

import costs
import gradients

from util import modifiers

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
        for y_batch, tx_batch in modifiers.batch_iter(y, tx, batch_size=1, num_batches=1):
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


def logistic_regression(y, tx, initial_w, max_iters, gamma, newton=False):
    """Logistic regression using GD or Newton's method."""

    def step_factor(w, grad):
        """Calculate the step-factor depending on whether we are running the Newton method."""
        return grad if not newton else np.linalg.solve(gradients.hessian(w, tx), grad)

    desc = 'LOG' if not newton else 'LOG-N'
    w = initial_w

    for n_iter in range(max_iters):
        grad = gradients.log_likelihood_gradient(y, tx, w)  # compute log-likelihood gradient
        w = w - gamma * step_factor(w, grad)  # compute new w
        loss = costs.compute_log_likelihood_error(y, tx, w)
        print("{desc}({bi}/{ti}): loss={l}".format(desc=desc, bi=n_iter, ti=max_iters-1, l=loss))
    
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, newton=False):
    """Regularised logistic regression using GD or Newton's method."""

    def step_factor(w, grad):
        """Calculate the step-factor depending on whether we are running the Newton method."""
        return grad

    def calc_cost(w):
        """Calculate log-likelihood error, adding the penalisation term."""
        return costs.compute_log_likelihood_error(y, tx, w) + (lambda_ / 2.0) * w.dot(w)

    desc = 'RLOG'
    w = initial_w

    thres = 1e-8
    previous_loss = 0.0

    for n_iter in range(max_iters):
        # compute penalised log likelihood gradient
        grad = gradients.log_likelihood_gradient(y, tx, w) + lambda_ * w
        w = w - gamma * step_factor(w, grad)  # compute new w

        loss = calc_cost(w)
        if np.abs(loss - previous_loss) < thres:
            break

        print("{desc}({bi}/{ti}): loss={l}".format(desc=desc, bi=n_iter, ti=max_iters-1, l=loss))
        previous_loss = loss  # update previous loss
    
    return w, loss
