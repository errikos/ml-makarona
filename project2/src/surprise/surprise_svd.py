import numpy as np
import csv
import sys

from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate

from helpers import load_data
from tune_grid_search import tune_grid_search

# Best params(?):
# factors = 10
# epochs = 400
# lr_all = 0.0002
# reg_all = 0.001
# rmse (with 0.2 test and training sets): ~1.045

# Load ratings
ratings_path = "./data/train_clean.csv"
reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
ratings = Dataset.load_from_file(ratings_path, reader)

# Test size
test_size = 0.1
seed = 50


def tune():
    print("Tuning...")

    # Sample random training set and test set.
    train_ratings, test_ratings = train_test_split(
        ratings, test_size=test_size, random_state=seed)

    best_param, best_rmse = -1, 100
    for param in range(1, 1000, 100):

        # Build SVD model.
        algorithm = SVD(
            n_factors=10, n_epochs=400, lr_all=0.0002, reg_all=param / 10000)

        # Train the algorithm on the training set, and predict ratings
        # for the test set.
        algorithm.fit(train_ratings)
        predictions = algorithm.test(test_ratings)

        # Then compute RMSE
        print("Reg:", param / 10000)
        rmse = accuracy.rmse(predictions)
        if rmse < best_rmse:
            best_rmse = rmse
            best_param = param

    print("Best reg:", best_param, " with rmse:", best_rmse)


def tune_gs():
    param_grid = {
        'n_factors': range(10, 50, 10),
        'n_epochs': [400],
        'lr_all': [x / 10000 for x in range(2, 10, 2)],
        'reg_all': [x / 1000 for x in range(1, 10, 2)]
    }

    tune_grid_search(
        ratings, SVD, param_grid, "svd,txt", n_jobs=2, pre_dispatch=4)


def test():
    print("Testing...")

    # Build SVD model.
    algorithm = SVD(n_factors=10, n_epochs=400, lr_all=0.0002, reg_all=0.001)

    # Sample random training set and test set.
    train_ratings, test_ratings = train_test_split(
        ratings, test_size=test_size, random_state=seed)

    # Train the algorithm on the training set, and predict ratings
    # for the test set.
    algorithm.fit(train_ratings)
    predictions = algorithm.test(test_ratings)

    # Then compute RMSE
    accuracy.rmse(predictions)


def test_crossval(cv=3):
    print("Cross validating...")

    # Build SVD model.
    algorithm = SVD(n_factors=10, n_epochs=400, lr_all=0.0001, reg_all=0.001)

    # Run 3-fold cross-validation and print results
    cross_validate(algorithm, ratings, measures=['RMSE'], cv=cv, verbose=True)


def submit():
    print("Creating submission...")

    factors = 10
    epochs = 400
    lr_all = 0.0001
    reg_all = 0.001

    # Retrieve the trainset.
    train_ratings = ratings.build_full_trainset()

    # Build SVD model and train it.
    sim_options = {'name': 'pearson', 'user_based': True}
    algorithm = SVD(
        n_factors=factors, n_epochs=epochs, lr_all=lr_all, reg_all=reg_all)
    algorithm.fit(train_ratings)

    # Get submission file format
    print("Producing submission file...")
    sample_submission_path = "../../data/submission.csv"
    test_ratings = load_data(sample_submission_path, sparse_matrix=False)

    rows, cols = np.nonzero(test_ratings)
    zp = list(zip(rows, cols))
    zp.sort(key=lambda tup: tup[1])

    # Create submission file
    submission_path = (
        "./submissions/surprise_svd_" + str(factors) + "_" + str(epochs) + "_"
        + str(lr_all) + "_" + str(reg_all) + ".csv")
    csvfile = open(submission_path, 'w')

    fieldnames = ['Id', 'Prediction']
    writer = csv.DictWriter(
        csvfile, delimiter=",", fieldnames=fieldnames, lineterminator='\n')
    writer.writeheader()

    counter = 0
    for row, col in zp:

        counter += 1
        if counter % 1000 == 0:
            print("Progress: %d/%d" % (counter, len(rows)))

        uid = str(row)
        iid = str(col)
        val = int(round(algorithm.predict(uid, iid)[3]))

        if val > 5:
            val = 5
        elif val < 1:
            val = 1

        r = "r" + str(row + 1)
        c = "c" + str(col + 1)
        writer.writerow({'Id': r + "_" + c, 'Prediction': val})

    csvfile.close()


if __name__ == '__main__':
    if len(sys.argv) == 2:
        if sys.argv[1] == '--tune':
            tune()
        elif sys.argv[1] == '--test':
            test()
        elif sys.argv[1] == '--crossval':
            test_crossval()
        elif sys.argv[1] == '--submit':
            submit()
        elif sys.argv[1] == '--tunegs':
            tune_gs()
        else:
            test()
    else:
        test()
