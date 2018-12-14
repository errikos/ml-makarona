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
from create_predictions import clean_predictions_to_file 


# Load ratings
ratings_path = "./data/train_clean.csv"
reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
ratings = Dataset.load_from_file(ratings_path, reader)

# Test parameters
test_size = 0.1
seed = 50

# Best params(?):
# n_epochs = 140
# n_cltr_u = 2
# n_cltr_i = 9

def tune():
    print("Tuning...")

    # Sample random training set and test set.
    train_ratings, test_ratings = train_test_split(
        ratings, test_size=test_size, random_state=seed)

    best_param, best_rmse = -1, 100
    for cltr_i in range(1, 11):

        # Build KNN item based model.
        algorithm = CoClustering(n_epochs=140, n_cltr_u=2, n_cltr_i=cltr_i)

        # Train the algorithm on the training set, and predict ratings
        # for the test set.
        algorithm.fit(train_ratings)
        predictions = algorithm.test(test_ratings)

        # Then compute RMSE
        print("cltr_i:", cltr_i)
        rmse = accuracy.rmse(predictions)
        if rmse < best_rmse:
            best_rmse = rmse
            best_param = cltr_i

    print("Best n_cltr_i:", best_param, " with rmse:", best_rmse)


def tune_gs():
    param_grid = {
        'n_epochs': range(100, 200, 10),
        'n_cltr_u': range(2, 10),
        'n_cltr_i': range(2, 10)
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
    algorithm = CoClustering(n_epochs=140, n_cltr_u=2, n_cltr_i=9)

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

    n_epochs=140 
    n_cltr_u=2 
    n_cltr_i=9

    # Retrieve the trainset.
    train_ratings = ratings.build_full_trainset()

    # Build KNN item based model and train it.
    algorithm = CoClustering(n_epochs=n_epochs, n_cltr_u=n_cltr_u, 
                             n_cltr_i=n_cltr_i)
    algorithm.fit(train_ratings)

    # Get submission file format
    print("Producing submission file...")
    sample_submission_path = "../../data/submission.csv"
    test_ratings = load_data(sample_submission_path, sparse_matrix=False)

    rows, cols = np.nonzero(test_ratings)
    zp = list(zip(rows, cols))
    zp.sort(key=lambda tup: tup[1])

    # Create submission file
    submission_path = "./submissions/surprise_coclustering_" + str(n_epochs) \
                         + "_" + str(n_cltr_u) + "_" + str(n_cltr_i) + ".csv"
                         
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
        elif sys.argv[1] == '--intermediate':
            clean_predictions_to_file("./data/988_test.csv", 
                                    "./data/",
                                    "./data/intermediate_988/interm_coclostering.csv",
                                    CoClustering()
        else:
            test()
    else:
        submit()
