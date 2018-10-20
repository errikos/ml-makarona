# -*- coding: utf-8 -*-
import abc
import numpy as np
import os

import implementations as impl
import costs
from util import loaders, parsers, testers
import costs


class Fitter(metaclass=abc.ABCMeta):

    def __init__(self, validation_param, degree=1, do_std=True, do_rm_samples=False,
                 do_rm_features=False, do_tune_hyper=False, do_cross_validate=False):
        self.validation_param = validation_param
        self.degree = degree
        self.do_std = do_std
        self.do_rm_samples = do_rm_samples
        self.do_rm_features = do_rm_features
        self.do_tune_hyper = do_tune_hyper
        self.do_cross_validate = do_cross_validate

    def _run(self, data_y, data_x, data_ids):
        raise NotImplementedError

    def _run_hyper(self, data_y, data_x, data_ids):
        raise NotImplementedError
        
    def run(self, data_y, data_x, data_ids):
        if self.do_rm_features:
            data_x = parsers.cut_features(data_x)
        if self.do_rm_samples:
            data_y, data_x = parsers.cut_samples(data_y, data_x)
        if self.do_std:
            data_x, mean, std = parsers.standardize(data_x)

        # TODO Move build_poly inside train and validate(?)
        # Build polynomial
        data_x = parsers.build_poly(data_x, self.degree, True)
        # self.mean, self.std = mean, std

        if self.do_tune_hyper:
            self._run_hyper(data_y, data_x, data_ids)
        else:
            self._run(data_y, data_x, data_ids)

    def _train_and_validate(self, data_y, data_x, data_ids, f, *args, ratio=0.7):
        # Split data
        # train_y, train_x, train_ids, lc_test_y, lc_test_x, lc_test_ids = \
        #     parsers.split_data_rand(data_y, data_x, data_ids, self.validation_param)
        train_y, train_x, lc_test_y, lc_test_x = parsers.split_data_rand(data_y, data_x,
                                                                         self.validation_param)
        # call train function
        w, err = f(train_y, train_x, *args)

        # TODO: Check if test error is correct and return it
        test_error = costs.compute_mse(lc_test_y, lc_test_x, w)

        # Predict labels for local testing data
        lc_pred_y = parsers.predict_labels(w, lc_test_x)

        matches = np.sum(lc_test_y == lc_pred_y)
        accuracy = matches / lc_test_y.shape[0]
        print('Accuracy:', accuracy)

        return w, test_error

    def _make_predictions(self, w):
        raise NotImplementedError

    def _train_and_cross_validate(self, data_y, data_x, data_ids, f, *args, k=4):
    
        # Create k subsets of the dataset
        subsets_y, subsets_x = parsers.k_fold_random_split(data_y, data_x, k, seed=1)

        # Train and validate k times, each time picking subset i as the test set
        averageTestError = 0
        averageAccuracy = 0
        for i in range(k):

            # Concatenate k-1 subsets 
            train_x = np.concatenate([subsets_x[j] for j in range(k) if j != i], 0)
            train_y = np.concatenate([subsets_y[j] for j in range(k) if j != i], 0)

            # call train function
            w, err = f(train_y, train_x, *args)

            # TODO Return average test error
            # Calculate test error, with subset i as test set
            averageTestError += costs.compute_mse(subsets_y[i], subsets_x[i], w)

            # Predict labels for local testing data
            lc_pred_y = parsers.predict_labels(w, subsets_x[i])

            matches = np.sum(subsets_y[i] == lc_pred_y)
            averageAccuracy += matches / subsets_y[i].shape[0]

        averageTestError /= k
        averageAccuracy /= k

        print("averageAccuracy: " + str(averageAccuracy))

        return w, averageTestError


class GDFitter(Fitter):
    """Fitter implementing linear regression using gradient descent."""

    def __init__(self, ratio, max_iter, gamma, **kwargs):
        super().__init__(ratio, **kwargs)
        self.max_iter = max_iter
        self.gamma = gamma

    def _run(self, data_y, data_x, data_ids):
        w_init = np.zeros((data_x.shape[1], ))
        f = impl.least_squares_GD
        args = [w_init, self.max_iter, self.gamma]
        if self.do_cross_validate:
            self._train_and_cross_validate(data_y, data_x, data_ids, f, *args)
        else:
            self._train_and_validate(data_y, data_x, data_ids, f, *args)

    def _run_hyper(self, data_y, data_x, data_ids):
        print("METHOD NOT SUPPORTED YET")
        raise NotImplementedError

    def _tune_gamma(self):
        for v in range(1, 20):
            yield v / 10
    
    def _make_predictions(self, w):
        test_path = os.path.join('..', 'data', 'test.csv')
        _, test_tx, test_ids = loaders.load_csv_data(test_path)

        if self.do_std:
            test_tx, _, _ = parsers.standardize(test_tx)

        test_tx = parsers.cut_features(test_tx) if self.do_rm_features else test_tx
        pred_y = parsers.predict_labels(w, parsers.build_poly(test_tx, self.degree, True))

        loaders.create_csv_submission(test_ids, pred_y, "gd.csv")