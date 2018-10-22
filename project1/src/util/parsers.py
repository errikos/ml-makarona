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

# # My version. Uses reshape
# def split_data_rand(y, x, ids, ratio, seed=1):
#     """Split data to training and testing set given the ratio."""
#     # print(x[1,:])
#     np.random.seed(seed)
#     #y = y.reshape(y.shape[0], 1)
#     allData = np.concatenate((y, x), axis=1)

#     np.random.shuffle(allData)

#     y = allData[:, 0]
#     x = allData[:, 1:]

#     # print(x[1,:])
#     train_x = x[0:np.floor(ratio * len(x)).astype(int)]
#     train_y = y[0:np.floor(ratio * len(x)).astype(int)]
#     train_ids = ids[0:np.floor(ratio * len(x)).astype(int)]

#     test_x = x[np.floor(ratio * len(x)).astype(int):len(x)]
#     test_y = y[np.floor(ratio * len(x)).astype(int):len(x)]
#     test_ids = ids[np.floor(ratio * len(x)).astype(int):len(x)]

#     return train_y, train_x, train_ids, test_y, test_x, test_ids


def split_data_rand(y, x, ratio, myseed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(myseed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return y_tr, x_tr, y_te, x_te 



def k_fold_random_split(y, x, k, seed=1):

    # Create space for k subsets of the initial dataset
    subsets_x = [None] * k
    subsets_y = [None] * k

    # Rearrange the indices of the initial dataset
    np.random.seed(seed)
    indices = np.random.permutation(len(y))

    # Calculate the number of rows per subset
    rows = np.floor(len(y) / k).astype(int)

    # Populate subsets
    for i in range(k - 1):
        subsets_x[i] = x[indices[i * rows : i * rows + rows]]
        subsets_y[i] = y[indices[i * rows : i * rows + rows]]

    subsets_x[k - 1] = x[indices[(k - 1) * rows :]]
    subsets_y[k - 1] = y[indices[(k - 1) * rows :]]

    return subsets_y, subsets_x

# My version: Calculates mean and std PER FEATURE without the -999s
# Sets all -999 to mean (0)
# Gets 0.74336 (linear regr)
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

# My version2: Calculates mean and std PER FEATURE without the -999s
# Sets all -999 to normal or uniform values with 0 mean and 1 std
# Gets 0.74124 (normal) (linear regr)
# Gets 0.74122 (uniform) (linear regr)
# def standardize(x):
#     """Standardise each feature. Returns mean of last feat."""
#     for i in range(x.shape[1]):
#         feature = x[:,i]
#         invalid = [feature == -999.0]
#         valid   = [feature != -999.0]
#         mean    = np.mean(feature[valid])
#         std     = np.std(feature[valid])
#         feature = (feature-mean)/std
#         #feature[invalid] = 0
#         #feature[invalid] = np.random.normal(0,1,feature[invalid].shape[0])
#         #feature[invalid] = np.random.uniform(-1,1,feature[invalid].shape[0])
#         x[:,i] = feature
#     return x, mean, std


# Their version: Calculates mean and std for the whole table with the -999
# Gets 0.68998 (linear regr)
# def standardize(x):
#     """Standardize the original data set."""
#     mean_x = np.mean(x)
#     x = x - mean_x
#     std_x = np.std(x)
#     x = x / std_x
#     return x, mean_x, std_x

# Their version: but per feature
# Gets 0.74334 (linear regr)
# def standardize(x):
#     """Standardize the original data set."""
#     for i in range(x.shape[1]):
#         feature = x[:,i]
#         mean_x = np.mean(feature)
#         feature = feature - mean_x
#         std_x = np.std(feature)
#         feature = feature / std_x
#         x[:,i] = feature
#     return x, mean_x, std_x


def sigmoid(z):
    """Return the sigmoid of x.
    x can be scalar or vector
    could remove one exp
    """
    sig = np.exp(z)/(np.exp(z) + 1)
    # print("Sigmoid is: ", sig, " (if high, causes overflow)")
    return sig

# Their Version
# def predict_labels(weights, data):
#     """Generates class predictions given weights, and a test data matrix"""
#     y_pred = np.dot(data, weights)
#     y_pred[np.where(y_pred <= 0)] = -1
#     y_pred[np.where(y_pred > 0)] = 1
    
#     return y_pred

def predict_labels(weights, data, is_logistic=False):
    """Generate class predictions given weights, and a test data matrix."""
    if is_logistic:
        y_pred = np.dot(data, weights)
        # TODO: REMEMBER TO CHECK IF SIGMOID HELPS
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
    return cut_y.reshape((cut_y.shape[0],)), cut_tx


def cut_features(some_tx):
    """Remove features that have at least on -999 value."""
    return np.delete(some_tx, [0, 4, 5, 6, 12, 23, 24, 25, 26, 27, 28], 1)
    # return np.delete(some_tx, [4,5,6,12,23,24,25,26,27,28], 1)
    # return np.delete(some_tx, [4,5,6,12,26,27,28], 1)
    # return np.delete(some_tx, [4], 1)
    # return some_tx