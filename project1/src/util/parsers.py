# -*- coding: utf-8 -*-
import itertools

import numpy as np
import implementations as impl
import costs

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Generate a minibatch iterator for a dataset.

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


def build_poly(tx, degree, do_add_bias=True, odd_only=False):
    """Polynomial basis functions for input data x, for j=0 up to j=degree."""
    _, D = tx.shape
    new_tx = np.zeros((tx.shape[0], degree * D))

    step = 2 if odd_only else 1

    j = 0
    for feat in range(0, D):
        for i in range(1, degree + 1, step):
            new_tx[:, j] = np.power(tx[:, feat], i)
            j = j + 1

    return np.concatenate((np.ones((tx.shape[0], 1)), new_tx), axis=1) if do_add_bias else new_tx


def split_data_rand(y, tx, ratio, seed=1):
    """Randomly split the dataset, based on the split ratio and the given seed."""
    np.random.seed(seed)

    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]

    # create and return splits
    return y[index_tr], tx[index_tr], y[index_te], tx[index_te]


def k_fold_random_split(y, tx, k, seed=1):
    """Create k random splits of the y and tx arrays, based on the given seed."""
    # Create space for k subsets of the initial dataset
    subsets_y = [None] * k
    subsets_tx = [None] * k

    # Rearrange the indices of the initial dataset
    np.random.seed(seed)
    indices = np.random.permutation(len(y))

    # Calculate the number of rows per subset
    rows = np.floor(len(y) / k).astype(int)

    # Populate subsets
    for i in range(k - 1):
        subsets_y[i] = y[indices[i * rows : i * rows + rows]]
        subsets_tx[i] = tx[indices[i * rows : i * rows + rows]]

    subsets_y[k - 1] = y[indices[(k - 1) * rows :]]
    subsets_tx[k - 1] = tx[indices[(k - 1) * rows :]]

    return subsets_y, subsets_tx


def eliminate_minus_999(tx):
    """Eliminates the -999 values per feature, by setting them to the feature's median."""
    for i in range(tx.shape[1]):
        feature = tx[:, i]
        invalid = feature == -999.0
        valid = feature != -999.0
        median = np.median(feature[valid])
        feature[invalid] = median
        tx[:, i] = feature
    return tx


def standardize(tx):
    """Standardise each feature.
    
    Standardises each feature by calculating the mean and standard deviation per feature,
    not taking into account the -999 values. Sets all -999 values to 0.0 (mean).
    """
    for i in range(1, tx.shape[1]):
        feature = tx[:, i]
        invalid = feature == -999.0
        valid = feature != -999.0
        mean = np.mean(feature[valid])
        std  = np.std(feature[valid])
        feature = (feature - mean) / std
        feature[invalid] = 0.0
        tx[:, i] = feature
    return tx


def build_poly_of_feature(x_i, degree):
    """Augment the given feature to the specified degree and return it."""
    richArray = np.zeros((x_i.shape[0], degree))

    for i in range(1, degree + 1):
        richArray[:, (i-1)] = np.power(x_i[:], i)

    return richArray


def augment_feature(tx, i, degree):
    """
    Augment the feature i of matrix tx to a polynomial of the given degree
    and return the modified matrix tx.
    """
    y = tx[:, :i]
    temp = build_poly_of_feature(tx[:, i], degree)
    y = np.concatenate((y, temp), axis=1)
    y = np.concatenate((y[:, :], tx[:, (i+1) :]), axis=1)
    return y


def build_poly_with_degrees(tx, degrees, do_add_bias=True):
    """Gets a dataset of features and an array of degrees to be applied
       to each feature, and augments features according to ther specified
       degree 
    """
    y = np.empty((tx.shape[0], 1))
    j = 0

    for i in range(tx.shape[1]):
        temp = build_poly_of_feature(tx[:, i], degrees[i])
        y = np.concatenate((y[:, :j], temp), axis=1)
        j = y.shape[1]
        y = np.concatenate((y[:, :], tx[:, (i+1) :]), axis=1)

    return np.concatenate((np.ones((tx.shape[0], 1)), y), axis=1) if do_add_bias else y


def predict_labels(weights, data, is_logistic=False):
    """Generate class predictions given weights, and a test data matrix."""
    y_pred = costs.sigmoid(data.dot(weights)) if is_logistic else data.dot(weights)
    cutoff, lower, upper = (0.5, 0, 1) if is_logistic else (0, -1, 1)

    y_pred[np.where(y_pred <= cutoff)] = lower
    y_pred[np.where(y_pred > cutoff)] = upper

    return y_pred


def drop_minus_999_features(tx):
    """Remove features that have at least one -999 value."""
    cols = [c for c in range(tx.shape[1]) if -999 in tx[:, c]]
    return np.delete(tx, cols, 1)


def augment_with_binary(tx):
    """Add a binary column for every group of columns containing -999 values."""
    _, D = tx.shape

    # construct groups of columns that have the -999 values in the same places
    groups = sorted([(tx[np.where(tx[:, i] == -999)].shape[0], i) for i in range(D)],
                    key=lambda x: x[0])
    groups = itertools.dropwhile(lambda x: x[0] == 0, groups)
    groups = [(group, [k[1] for k in keys])
              for group, keys in itertools.groupby(groups, key=lambda x: x[0])]
    # groups is a list of pairs: first element is -999 count, second element is a list of columns

    # create binary columns for each group, concatenate with tx and return
    binary_cols = np.array([tx[:, cols[0]] != -999 for _, cols in groups]).T.astype(int)
    return np.concatenate((tx, binary_cols), axis=1)
