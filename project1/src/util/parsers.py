# -*- coding: utf-8 -*-

import numpy as np

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def build_poly(x, degree, do_add_bias):
    """Polynomial basis functions for input data x, for j=0 up to j=degree.
    Important: Adds bias column.
    """
    numOfFeat = x.shape[1]
    richArray = np.zeros((x.shape[0], degree * numOfFeat))
    j = 0

    for feat in range(0, numOfFeat):

        for i in range(1, degree + 1):
            richArray[:, j] = np.power(x[:, feat], i)
            j = j + 1

    if(do_add_bias):
        zeroethPower = np.ones((x.shape[0], 1))
        res = np.concatenate((zeroethPower, richArray), axis=1)
    else:
        res = richArray
    return res


def split_data(x, y, ids, ratio, seed=1):
    """Split data determninistically."""
    np.random.seed(seed)

    train_x = x[0:np.floor(ratio * len(x)).astype(int)]
    train_y = y[0:np.floor(ratio * len(x)).astype(int)]
    train_ids = y[0:np.floor(ratio * len(x)).astype(int)]

    test_x = x[np.floor(ratio * len(x)).astype(int):len(x)]
    test_y = y[np.floor(ratio * len(x)).astype(int):len(x)]
    test_ids = y[np.floor(ratio * len(x)).astype(int):len(x)]

    return train_x, train_y, train_ids, test_x, test_y, test_ids


def split_data_rand(x, y, ids, ratio, seed=1):
    """Split data to training and testing set given the ratio."""
    # print(x[1,:])
    np.random.seed(seed)
    y = y.reshape(y.shape[0], 1)
    allData = np.concatenate((y, x), axis=1)

    np.random.shuffle(allData)

    y = allData[:, 0]
    x = allData[:, 1:]

    # print(x[1,:])
    train_x = x[0:np.floor(ratio * len(x)).astype(int)]
    train_y = y[0:np.floor(ratio * len(x)).astype(int)]
    train_ids = y[0:np.floor(ratio * len(x)).astype(int)]

    test_x = x[np.floor(ratio * len(x)).astype(int):len(x)]
    test_y = y[np.floor(ratio * len(x)).astype(int):len(x)]
    test_ids = y[np.floor(ratio * len(x)).astype(int):len(x)]

    return train_x, train_y, train_ids, test_x, test_y, test_ids


def standardize(x):
    """Standardise each feature. Returns mean of last feat."""

    for i  in range(x.shape[1]):

        feature = x[:,i]
        invalid = [feature == -999.0]
        valid   = [feature != -999.0]

        mean    = np.mean(feature[valid])
        std     = np.std(feature[valid])

        feature = (feature-mean)/std

        feature[invalid] = 0
        x[:,i] = feature

    return x, mean, std


# def standardize(x):
#     """Standardize the original data set."""
#     mean_x = np.mean(x)
#     # print("MEANS ARE: ", mean_x)
#     x = x - mean_x
#     std_x = np.std(x)
#     # print("STDS ARE: ", std_x)
#     x = x / std_x
#     return x, mean_x, std_x


def sigmoid(z):
    """Return the sigmoid of x.
    x can be scalar or vector
    could remove one exp
    """
    sig = np.exp(z)/(np.exp(z) + 1)
    # print("Sigmoid is: ", sig, " (if high, causes overflow)")
    return sig


def predict_labels(weights, data, is_logistic):
    """Generate class predictions given weights, and a test data matrix."""
    if(is_logistic):
        y_pred = np.dot(data, weights)
        #y_pred = sigmoid(np.dot(data, weights))
        y_pred[np.where(y_pred <= 0.5)] = 0
        y_pred[np.where(y_pred > 0.5)] = 1
    else:
        y_pred = np.dot(data, weights)
        y_pred[np.where(y_pred <= 0)] = -1
        y_pred[np.where(y_pred > 0)] = 1

    return y_pred


def cut_samples(some_y, some_tx):
    """Remove all samples that have at least one -999 value."""
    some_y = some_y.reshape(some_y.shape[0], 1)
    allData = np.concatenate((some_y, some_tx), axis=1)
    badRows, badCols = np.where(allData == -999)
    uniqueBadRows = np.unique(badRows)

    allData = np.delete(allData, uniqueBadRows, 0)
    cut_y = allData[:, 0]
    cut_tx = allData[:, 1:]
    return cut_y, cut_tx


def cut_features(some_tx):
    """Remove features that have at least on -999 value."""
    return np.delete(some_tx, [0, 4, 5, 6, 12, 23, 24, 25, 26, 27, 28], 1)
    # return np.delete(some_tx, [4,5,6,12,23,24,25,26,27,28], 1)
    # return np.delete(some_tx, [4,5,6,12,26,27,28], 1)
    # return np.delete(some_tx, [4], 1)
    # return some_tx