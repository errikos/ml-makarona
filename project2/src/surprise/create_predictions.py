import numpy as np
import csv
import sys

from surprise import KNNBaseline
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import cross_validate

from helpers import load_data, calculate_rmse, write_clean, load_clean
from tune_grid_search import tune_grid_search


def clean_predictions_to_file(sample_submission_path, input_path, output_path, algo):

    # Retrieve the trainset.
    reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
    train_data = Dataset.load_from_file(input_path+"988_training.csv", reader)
    #train_data = load_clean(input_path+"988_training.csv")
    test_data = load_clean(input_path+"988_test.csv")
    # Build KNN item based model and train it.
    algorithm = algo
    train_ratings = train_data.build_full_trainset()
    algorithm.fit(train_ratings)

    # Get submission file format
    print("Producing clean intermediate file...")
    sample_submission_path = "../../data/submission.csv"

    rows, cols = np.nonzero(test_data)
    zp = list(zip(rows, cols))

    # Create submission file
    #submission_path = "./submissions/surprise_item_based_bsln_top" +
                       # str(K) +"_full_enhanced.csv"
    # csvfile = open(output_path, 'w')

    # fieldnames = ['Id', 'Prediction']
    # writer = csv.DictWriter(csvfile, delimiter=",",
    #                     fieldnames=fieldnames, lineterminator = '\n')
    # writer.writeheader()

    data_to_write = [None] * len(zp)
    for i, row_col in enumerate(zp):

        uid = str(row_col[0])
        iid = str(row_col[1])
        val = algorithm.predict(uid, iid)[3]

        if val > 5.0:
            val = 5.0
        elif val < 1.0:
            val = 1.0
        data_to_write[i] = (row_col[0], row_col[1], val)
        
            
    write_clean(data_to_write, output_path)
