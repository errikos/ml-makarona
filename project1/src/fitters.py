# -*- coding: utf-8 -*-
import abc
import numpy as np
import os

import implementations as impl
from util import loaders, parsers, testers


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
            # for i  in range(x.shape[1]): IndexError: tuple index out of range
            data_x, mean, std = parsers.standardize(data_x)
        # Build polynomial
        data_x = parsers.build_poly(data_x, self.degree, True)
        # self.data_x, self.data_y, self.data_ids = data_x, data_y, data_ids 
        self.mean, self.std = mean, std

        if self.do_tune_hyper:
            self._run_hyper(data_y, data_x, data_ids)
        else:
            self._run(data_y, data_x, data_ids)

    def _train_and_validate(self, data_y, data_x, data_ids, f, *args, ratio=0.7):
        # Split data
        train_y, train_x, train_ids, lc_test_y, lc_test_x, lc_test_ids = \
            parsers.split_data_rand(data_y, data_x, data_ids, self.validation_param)
        # set initial w
        w_init = np.zeros((train_x.shape[1], ))
        # call train function
        w, err = f(train_y, train_x, w_init, *args)

        # Predict labels for local testing data
        lc_pred_y = parsers.predict_labels(w, lc_test_x)

        # TODO: Also return test error

        matches = np.sum(lc_test_y == lc_pred_y)
        accuracy = matches / lc_test_y.shape[0]

        # -----------------------------------------------------------------------------------------
        # TODO: Move to each subclass
        test_path = os.path.join('..', 'data', 'test.csv')
        test_y, test_tx, test_ids = loaders.load_csv_data(test_path)

        if self.do_std:
            test_tx, mean, std = parsers.standardize(test_tx)

        test_tx = parsers.cut_features(test_tx) if self.do_rm_features else test_tx
        pred_y = parsers.predict_labels(w, parsers.build_poly(test_tx, self.degree, True), False)

        loaders.create_csv_submission(test_ids, pred_y, "GD.csv")

    def _train_and_cross_validate(self, data_x, data_y, data_ids, f, *args, k=4):
        pass


class GD_fitter(Fitter):

    def __init__(self, ratio, max_iter, gamma, **kwargs):
        super().__init__(ratio, **kwargs)
        self.max_iter = max_iter
        self.gamma = gamma

    def _run(self, data_y, data_x, data_ids):
        f = impl.least_squares_GD
        args = [self.max_iter, self.gamma]
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