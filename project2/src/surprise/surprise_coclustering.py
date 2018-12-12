import numpy as np
import csv
import sys

from surprise import CoClustering
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate

from helpers import load_data
from tune_grid_search import tune_grid_search

# Load ratings
ratings_path = "./data/train_clean.csv"
reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
ratings = Dataset.load_from_file(ratings_path, reader)

# Test parameters
test_size = 0.1
seed = 50


def tune():
    print("Tuning...")

    # Sample random training set and test set.
    train_ratings, test_ratings = train_test_split(
        ratings, test_size=test_size, random_state=seed)

    best_param, best_rmse = -1, 100
    for epochs in range(20, 100, 20):

        # Build KNN item based model.
        algorithm = CoClustering(n_epochs=epochs)

        # Train the algorithm on the training set, and predict ratings
        # for the test set.
        algorithm.fit(train_ratings)
        predictions = algorithm.test(test_ratings)

        # Then compute RMSE
        print("epochs:", epochs)
        rmse = accuracy.rmse(predictions)
        if rmse < best_rmse:
            best_rmse = rmse
            best_param = epochs

    print("Best number of epochs:", best_param, " with rmse:", best_rmse)


def tune_gs():
    param_grid = {
        'n_epochs': range(10, 100, 10),
        'n_cltr_u': range(1, 10),
        'n_cltr_i': range(1, 10)
    }

    tune_grid_search(
        ratings,
        CoClustering,
        param_grid,
        "coclustering.txt",
        n_jobs=2,
        pre_dispatch=2)


def test():
    print("Testing...")

    # Build KNN item based model.
    algorithm = CoClustering()

    # Sample random training set and test set.
    train_ratings, test_ratings = train_test_split(
        ratings, test_size=test_size, random_state=seed)

    # Train the algorithm on the training set, and predict ratings
    # for the test set.
    algorithm.fit(train_ratings)
    predictions = algorithm.test(test_ratings)

    # Then compute RMSE
    accuracy.rmse(predictions)


def test_crossval(cv=2):
    print("Cross validating...")

    # Build KNN item based model.
    algorithm = CoClustering()

    # Run 2-fold cross-validation and print results
    cross_validate(algorithm, ratings, measures=['RMSE'], cv=cv, verbose=True)


def submit():
    print("Creating submission...")

    # Retrieve the trainset.
    train_ratings = ratings.build_full_trainset()

    # Build KNN item based model and train it.
    algorithm = CoClustering()
    algorithm.fit(train_ratings)

    # Get submission file format
    print("Producing submission file...")
    sample_submission_path = "../../data/submission.csv"
    test_ratings = load_data(sample_submission_path, sparse_matrix=False)

    rows, cols = np.nonzero(test_ratings)
    zp = list(zip(rows, cols))
    zp.sort(key=lambda tup: tup[1])

    # Create submission file
    submission_path = "./submissions/surprise_coclustering.csv"
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
            test_crossval(2)
        elif sys.argv[1] == '--submit':
            submit()
        elif sys.argv[1] == '--tunegs':
            tune_gs()
        else:
            test()
    else:
        tune()
