import numpy as np
import csv
import sys

from surprise import KNNBaseline
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate

from helpers import load_data, calculate_rmse, load_clean_vec
from tune_grid_search import tune_grid_search


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""
    N, D = tx.shape
    lambda_ = 2 * N * lambda_

    a = tx.T.dot(tx) + lambda_ * np.eye(D)
    b = tx.T.dot(y)

    w = np.linalg.solve(a, b)
    loss = costs.compute_rmse(y, tx, w)
    return w, loss

test_ratings_path = "./data/988_test.csv"
# ratings_path = "./data/surprise_item_based_bsln_top50_full_enhanced_clean.csv"
users, items, test_ratings = load_clean_vec(test_ratings_path)

results_file_names = []

