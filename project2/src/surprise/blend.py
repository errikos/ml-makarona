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
    #loss = costs.compute_rmse(y, tx, w)
    return w #loss

def calculate_rmse(real_label, prediction):
    """calculate MSE."""
    t = real_label - np.round(prediction)
    return np.sqrt(1.0 * t.dot(t.T) / real_label.shape[0])
    #return 1.0 * t.dot(t.T)

test_ratings_path = "./data/988_test.csv"
# ratings_path = "./data/surprise_item_based_bsln_top50_full_enhanced_clean.csv"
test_users, test_items, test_ratings = load_clean_vec(test_ratings_path)

offset_path = "./data/intermediate_988/"
results_file_names = ["interm_item_based.csv",
                      "interm_user_based.csv",
                      "interm_slope.csv",
                      "svd.csv",
                      "interm_user_based_bsl.csv",
                      "interm_item_based_bsl.csv"]

interm_res_vecs = [None] * len(results_file_names)
for i, file_name in enumerate(results_file_names):
    usr, itm, rtings = load_clean_vec(offset_path + file_name)
    #print(usr[:10], itm[:10], rtings[:10])
    #interm_res_vecs[i] = np.round(rtings)
    interm_res_vecs[i] = rtings

# y is Nx1 (N is the number of ratings)
y = np.array(test_ratings).reshape((len(test_ratings),1))
# X is NxD (D is the number of models)
X = np.array(interm_res_vecs).transpose()
# w is Dx1
w = np.ones((X.shape[1],1)) * 1/X.shape[1]
# Ridge is: w* = (XT*X + lambda*I)^-1*XT*y
#print(y.shape, "\n", y[:,:10], "\n\n", X.shape, "\n", X[:10,:], "\n\n",w.shape, "\n", w)

w_star = ridge_regression(y, X, 0.001)
print("W* \n",w_star)

final_prediction = X.dot(w)
final_prediction = np.round(final_prediction.reshape(final_prediction.shape[0]))
#print("final prediction\n", final_prediction[:,0])

# Rmse of models separately
for i, model_pred in enumerate(interm_res_vecs):
    print("rmse is: ", results_file_names[i], "\n", calculate_rmse(test_ratings, model_pred))

print("SUM OF Weights: ", np.sum(w_star))
# Rmse of blend
print("Blend RMSE: ", calculate_rmse(test_ratings, final_prediction))


# Now produce predictions using the individual model predictions * their weights (based on the
# croudAI submission file) -------------------------------------------------------------------
 