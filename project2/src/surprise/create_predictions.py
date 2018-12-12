import numpy as np
import csv
import sys

from surprise import KNNBaseline
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate

from helpers import load_data, calculate_rmse
from tune_grid_search import tune_grid_search


def predictions_to_file(preditction_indices_path, output_path, algo):
    print("Creating prediciton...")

    # Retrieve the trainset.
    train_ratings = ratings.build_full_trainset()

    # Build KNN item based model and train it.
    algorithm = algo
    algorithm.fit(train_ratings)

    # Get submission file format
    print("Producing submission file...")
    #sample_submission_path = "../../data/submission.csv"
    test_ratings = load_data(preditction_indices_path, sparse_matrix=False)

    rows, cols = np.nonzero(test_ratings)
    zp = list(zip(rows, cols))
    zp.sort(key = lambda tup: tup[1])

    # Create submission file
    #submission_path = "./submissions/surprise_item_based_bsln_top" +
                        str(K) +"_full_enhanced.csv"
    csvfile = open(output_path, 'w')

    fieldnames = ['Id', 'Prediction']
    writer = csv.DictWriter(csvfile, delimiter=",",
                        fieldnames=fieldnames, lineterminator = '\n')
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