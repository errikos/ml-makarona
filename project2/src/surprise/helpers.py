# -*- coding: utf-8 -*-
"""some functions for help."""

from itertools import groupby

import numpy as np
import scipy.sparse as sp
import csv
import math


def read_txt(path):
    """
        read text file from path.
    """
    with open(path, "r") as f:
        return f.read().splitlines()


def load_data(path_dataset, sparse_matrix=True):
    """
        Load data in text format, one rating per line, and store in a
        sparse matrix or np.array (shape users x items)
    """
    data = read_txt(path_dataset)[1:]
    return preprocess_data(data, sparse_matrix)


def preprocess_data(data, sparse_matrix=True):
    """
        Returns an array of ratings with shape users x items
    """

    def deal_line(line):
        pos, rating = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row), int(col), float(rating)

    def statistics(data):
        row = set([line[0] for line in data])
        col = set([line[1] for line in data])
        return min(row), max(row), min(col), max(col)

    # parse each line
    data = [deal_line(line) for line in data]

    # do statistics on the dataset.
    min_row, max_row, min_col, max_col = statistics(data)

    # build rating matrix.
    if sparse_matrix:
        ratings = sp.lil_matrix((max_row, max_col))
    else:
        ratings = np.zeros((max_row, max_col))

    for row, col, rating in data:
        ratings[row - 1, col - 1] = rating
    return ratings


def load_clean(path_dataset):
    def deal_line(line):
        row, col, rating = line.split(',')
        return int(row), int(col), float(rating)

    def statistics(data):
        row = set([line[0] for line in data])
        col = set([line[1] for line in data])
        return min(row), max(row), min(col), max(col)

    data = read_txt(path_dataset)[1:]

    # Parse each line.
    data = [deal_line(line) for line in data]

    # Do statistics on the dataset.
    min_row, max_row, min_col, max_col = statistics(data)

    # Build rating matrix.
    ratings = np.zeros((max_row, max_col))

    for row, col, rating in data:
        ratings[row - 1, col - 1] = rating
    return ratings


def load_clean_vec(path_dataset):

    def deal_line(line):
        row, col, rating = line.split(',')
        return int(row), int(col), float(rating)


    data = read_txt(path_dataset)[1:]

    # Parse each line.
    data = [deal_line(line) for line in data]

    # Build rating matrix.
    users = np.zeros(len(data))
    items = np.zeros(len(data))
    ratings = np.zeros(len(data))

    for i, tup in enumerate(data):
        users[i] = tup[0]
        items[i] = tup[1]
        ratings[i] = tup[2]
    return users, items, ratings


def split_data(p_test=0.1, seed=988):
    """
        Load and split original ratings to training and test sets and store 
        them in two separate files (as user-item ratings).
    """
    orig_ratings_path = "../../data/train.csv"
    ratings = load_data(orig_ratings_path, sparse_matrix=False)

    # Set seed
    np.random.seed(seed)

    # Get non zero users and items
    nz_users, nz_items = np.nonzero(ratings)

    # Split the data and store training and test ratings in two \
    # separate files
    trainfile = open("./data/" + str(seed) + "_training.csv", 'w')
    testfile = open("./data/" + str(seed) + "_test.csv", 'w')

    fieldnames = ['User', 'Item', 'Rating']
    trainwriter = csv.DictWriter(trainfile, delimiter=",",
                                fieldnames=fieldnames, lineterminator='\n')
    testwriter = csv.DictWriter(testfile, delimiter=",",
                                fieldnames=fieldnames, lineterminator='\n')
    trainwriter.writeheader()
    testwriter.writeheader()

    # Randomly select a subset of items for each user, to go to the
    # test set.
    for user in set(nz_users):

        cols = np.nonzero(ratings[user, :])[0]

        selects = np.random.choice(cols, size=int(len(cols) * p_test))
        residual = list(set(cols) - set(selects))

        # Add to training set
        for item in residual:
            trainwriter.writerow({'User': user, 'Item': item,
                                  'Rating': ratings[user, item]})

        # Add to test set
        for item in selects:
            testwriter.writerow({'User': user, 'Item': item,
                                 'Rating': ratings[user, item]})

    trainfile.close()
    testfile.close()


def calculate_rmse(algorithm, true_ratings_path):
    """ 
        Compute the loss (RMSE) of the prediction of nonzero elements.
    """

    # Get submission file format
    true_ratings = load_clean(true_ratings_path)

    rows, cols = np.nonzero(true_ratings)
    zp = list(zip(rows, cols))

    mse = 0
    for row, col in zp:

        uid = str(row)
        iid = str(col)
        val = int(round(algorithm.predict(uid, iid)[3]))

        if val > 5:
            val = 5
        elif val < 1:
            val = 1

        mse += (true_ratings[row, col] - val)**2

    return math.sqrt(1.0 * mse / len(rows))


if __name__ == '__main__':
    split_data()
