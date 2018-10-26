import numpy as np
from costs import sigmoid

def mse_gradient(y, tx, w):
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def hessian(w, tx):
    sigm = sigmoid(tx.dot(w))
    S = sigm * (1 - sigm)
    return np.multiply(tx.T, S) @ tx


def log_likelihood_gradient(y, tx, w):
    s = sigmoid(tx.dot(w))
    return tx.T.dot(s - y)
