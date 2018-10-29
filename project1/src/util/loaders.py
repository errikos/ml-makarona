# -*- coding: utf-8 -*-

import numpy as np
import csv


def load_csv_data(data_path, sub_sample=False, is_logistic=False):
    """Load data and return y (class labels), tX (features) and ids.
    If the data is to be used by a logistic method, the labels should be (0,1).
    """
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1) or (0,1) if logistic
    neg_label = 0 if is_logistic else -1
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = neg_label

    # sub-sample
    if sub_sample:
        yb = yb[0:100]
        input_data = input_data[0:100, :]
        ids = ids[0:100]

    return yb, input_data, ids


def create_csv_submission(ids, y_pred, name):
    """Create an output file in csv format for submission to kaggle.

    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """

    y_pred[np.where(y_pred == 0)] = -1

    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames, lineterminator = '\n')
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})
