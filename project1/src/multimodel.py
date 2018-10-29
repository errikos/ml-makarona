#!/usr/bin/env python3
import os
import numpy as np

from util import modifiers, loaders
from implementations import ridge_regression


DEFAULT_DATA_PATH = os.path.join('..', 'data')


def _load_data(data_path, is_logistic=False):
    train_y, train_tx, train_ids = loaders.load_csv_data(os.path.join(data_path, 'train.csv'),
                                                         is_logistic=is_logistic)
    _, test_tx, test_ids = loaders.load_csv_data(os.path.join(data_path, 'test.csv'),
                                                 is_logistic=is_logistic)
    return train_y, train_tx, train_ids, test_tx, test_ids


def _split_train_dataset(y, tx, jet_num_idx=22):
    """Split the given training dataset into three distinct datasets.

    Datasets are split depending on the value of the 'PRI_jet_num' column,
    since this column dictates the -999 values for all the columns containing
    the latter (except for the 'DER_mass_MMC' column at index 0).
    """
    jet0_tx = tx[tx[:, jet_num_idx] == 0]
    jet0_tx = np.delete(jet0_tx, [jet_num_idx, 4, 5, 6, 12, 23, 24, 25, 26, 27, 28, 29], axis=1)
    jet0_y = y[tx[:, jet_num_idx] == 0]

    jet1_tx = tx[tx[:, jet_num_idx] == 1]
    jet1_tx = np.delete(jet1_tx, [jet_num_idx, 4, 5, 6, 12, 26, 27, 28], axis=1)
    jet1_y = y[tx[:, jet_num_idx] == 1]

    jetR_tx = tx[tx[:, jet_num_idx] >= 2]
    jetR_y = y[tx[:, jet_num_idx] >= 2]

    return jet0_y, jet1_y, jetR_y, jet0_tx, jet1_tx, jetR_tx


def _split_test_dataset(tx, jet_num_idx=22):
    """Same as _split_train_dataset, but for the testing dataset."""
    jet0_tx = tx[tx[:, jet_num_idx] == 0]
    jet0_tx = np.delete(jet0_tx, [jet_num_idx, 4, 5, 6, 12, 23, 24, 25, 26, 27, 28, 29], axis=1)

    jet1_tx = tx[tx[:, jet_num_idx] == 1]
    jet1_tx = np.delete(jet1_tx, [jet_num_idx, 4, 5, 6, 12, 26, 27, 28], axis=1)

    jetR_tx = tx[tx[:, jet_num_idx] >= 2]

    return jet0_tx, jet1_tx, jetR_tx


def _build_polynomials(degrees, *txs):
    return (modifiers.build_poly(tx, degree=d) for tx, d in zip(txs, degrees))


def run_multimodel():
    train_y, train_tx, _, test_tx, test_ids = _load_data(DEFAULT_DATA_PATH)
    N, D = train_tx.shape

    (train_jet0_y, train_jet1_y, train_jetR_y,
     train_jet0_tx, train_jet1_tx, train_jetR_tx) = _split_train_dataset(train_y, train_tx)

    degrees = (7, 10, 10)
    # build train polynomials
    train_jet0_tx, train_jet1_tx, train_jetR_tx = _build_polynomials(
        degrees, train_jet0_tx, train_jet1_tx, train_jetR_tx)

    ratio = 0.8
    seed = 155
    # split data to train and test
    train_jet0_y, train_jet0_tx, test_jet0_y, test_jet0_tx = modifiers.split_data_rand(
        train_jet0_y, train_jet0_tx, ratio, seed)
    train_jet1_y, train_jet1_tx, test_jet1_y, test_jet1_tx = modifiers.split_data_rand(
        train_jet1_y, train_jet1_tx, ratio, seed)
    train_jetR_y, train_jetR_tx, test_jetR_y, test_jetR_tx = modifiers.split_data_rand(
        train_jetR_y, train_jetR_tx, ratio, seed)

    print('Jet 0 shape:', *train_jet0_tx.shape)
    print('Jet 1 shape:', *train_jet1_tx.shape)
    print('Jet R shape:', *train_jetR_tx.shape)

    lambdas = (0.2, 0.1, 0.2)
    # train three models, one for each Jet group
    w0, loss0 = ridge_regression(train_jet0_y, train_jet0_tx, lambdas[0])
    w1, loss1 = ridge_regression(train_jet1_y, train_jet1_tx, lambdas[1])
    wR, lossR = ridge_regression(train_jetR_y, train_jetR_tx, lambdas[2])

    print('Losses per Jet:', round(loss0, 4), round(loss1, 4), round(lossR, 4))

    # get local testing accuracy
    train_y_preds = modifiers.predict_labels(w0, train_jet0_tx)

    tr_acc = []
    te_acc = []
    print('-------------------------------------------------------------')
    for w, tr_y, tr_x, te_y, te_x in zip([w0, w1, wR],
                                         [train_jet0_y, train_jet1_y, train_jetR_y],
                                         [train_jet0_tx, train_jet1_tx, train_jetR_tx],
                                         [test_jet0_y, test_jet1_y, test_jetR_y],
                                         [test_jet0_tx, test_jet1_tx, test_jetR_tx]):
        train_y_preds = modifiers.predict_labels(w, tr_x)
        matches_tr = np.sum(tr_y == train_y_preds)
        accuracy_tr = matches_tr / tr_y.shape[0]
        tr_acc.append(accuracy_tr)
        print('Training Accuracy:', round(accuracy_tr, 4))

        test_y_preds = modifiers.predict_labels(w, te_x)
        matches_te = np.sum(te_y == test_y_preds)
        accuracy_te = matches_te / te_y.shape[0]
        te_acc.append(accuracy_te)
        print('Testing Accuracy:', round(accuracy_te, 4))


    print('-------------------------------------------------------------')
    print('Weighted average TR:', round((len(train_jet0_tx) / (N * ratio) * tr_acc[0]) + 
                                        (len(train_jet1_tx) / (N * ratio) * tr_acc[1]) +
                                        (len(train_jetR_tx) / (N * ratio) * tr_acc[2]), 4))
    print('Weighted average TE:', round((len(test_jet0_tx) / (N * (1-ratio)) * te_acc[0]) + 
                                        (len(test_jet1_tx) / (N * (1-ratio)) * te_acc[1]) +
                                        (len(test_jetR_tx) / (N * (1-ratio)) * te_acc[2]), 4))
    
    # append test_ids to test_tx, needed for matching predictions later
    test_ids_re = np.reshape(test_ids, (test_ids.shape[0], 1))
    test_tx = np.concatenate((test_tx, test_ids_re), axis=1)

    # split test data
    test_jet0_tx, test_jet1_tx, test_jetR_tx = _split_test_dataset(test_tx)

    # save and detach test IDs for each jet
    test_jet0_tx_ids = test_jet0_tx[:, -1]
    test_jet0_tx = np.delete(test_jet0_tx, test_jet0_tx.shape[1]-1, axis=1)

    test_jet1_tx_ids = test_jet1_tx[:, -1]
    test_jet1_tx = np.delete(test_jet1_tx, test_jet1_tx.shape[1]-1, axis=1)

    test_jetR_tx_ids = test_jetR_tx[:, -1]
    test_jetR_tx = np.delete(test_jetR_tx, test_jetR_tx.shape[1]-1, axis=1)

    # build test polynomials
    test_jet0_tx, test_jet1_tx, test_jetR_tx = _build_polynomials(degrees, test_jet0_tx, test_jet1_tx, test_jetR_tx)

    # make predictions and append IDs for each jet
    pred_jet0 = modifiers.predict_labels(w0, test_jet0_tx)
    pred_jet0 = np.array(list(zip(pred_jet0, test_jet0_tx_ids)))

    pred_jet1 = modifiers.predict_labels(w1, test_jet1_tx)
    pred_jet1 = np.array(list(zip(pred_jet1, test_jet1_tx_ids)))

    pred_jetR = modifiers.predict_labels(wR, test_jetR_tx)
    pred_jetR = np.array(list(zip(pred_jetR, test_jetR_tx_ids)))

    # concatenate results and sort based on ID
    all_predictions = np.concatenate((pred_jet0, pred_jet1, pred_jetR), axis=0)
    all_predictions = all_predictions[all_predictions[:, 1].argsort()]
    all_predictions = np.delete(all_predictions, [1], axis=1)

    # create submission CSV
    loaders.create_csv_submission(test_ids, np.squeeze(all_predictions), 'smartass.csv')


def main():
    run_multimodel()

if __name__ == '__main__':
    main()