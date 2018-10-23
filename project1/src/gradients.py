import numpy as np
from costs import sigmoid

def compute_gradient(y, tx, w):
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def hessian(w, tx):
    def diag_form(xn):
        s = sigmoid(xn.T.dot(w))
        return s * (1 - s)
    S = np.diag(np.array([diag_form(sample) for sample in tx]))
    return tx.T.dot(S).dot(tx)


def log_likelihood_gradient(y, tx, w):
    diff = sigmoid(tx.dot(w)) - y
    return tx.T.dot(diff)