# -*- coding: utf-8 -*-

import numpy as np
import csv

# def load_data():
#     """Load data."""
#     data = np.loadtxt("dataEx3.csv", delimiter=",", skiprows=1, unpack=True)
#     x = data[0]
#     y = data[1]
#     return x, y


# def load_data_from_ex02(sub_sample=True, add_outlier=False):
#     """Load data and convert it to the metric system."""
#     path_dataset = "height_weight_genders.csv"
#     data = np.genfromtxt(
#         path_dataset, delimiter=",", skip_header=1, usecols=[1, 2])
#     height = data[:, 0]
#     weight = data[:, 1]
#     gender = np.genfromtxt(
#         path_dataset, delimiter=",", skip_header=1, usecols=[0],
#         converters={0: lambda x: 0 if b"Male" in x else 1})
#     # Convert to metric system
#     height *= 0.025
#     weight *= 0.454

#     # sub-sample
#     if sub_sample:
#         height = height[::50]
#         weight = weight[::50]

#     if add_outlier:
#         # outlier experiment
#         height = np.concatenate([height, [1.1, 1.2]])
#         weight = np.concatenate([weight, [51.5/0.454, 55.3/0.454]])

#     return height, weight, gender


def load_csv_data(data_path, sub_sample=False, is_logistic=False):
    """Load data and return y (class labels), tX (features) and ids.
    If the data is to be used by a logistic method, the labels should be (0,1).
    """
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1) or (0,1) if logistic
    yb = np.ones(len(y))
    if (is_logistic):
        yb[np.where(y == 'b')] = 0
    else:
        yb[np.where(y == 'b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[0:100]
        input_data = input_data[0:100, :]
        ids = ids[0:100]

    return yb, input_data, ids

# dark code
def build_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form."""
    print("Build model data is actually called")
    y = weight
    x = height
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx
    

def create_csv_submission(ids, y_pred, name):
    """Create an output file in csv format for submission to kaggle.
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames, lineterminator = '\n')
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})