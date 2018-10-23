# -*- coding: utf-8 -*-
import abc
import numpy as np
import itertools
import os

import implementations as impl
import costs
from util import loaders, parsers, testers
import costs


def print_dict(d, delim='='):
    for k, v in d.items():
        print('{k}{d}{v}'.format(k=k, d=delim, v=v), end=' ')
    print()


class Fitter(metaclass=abc.ABCMeta):
    """Base class for Fitter objects.

    Each fitter encapsulates a fitter behaviour and provides a nice interface
    for settings parameters, tuning hyper-parameters, etc.

    The core method implementations are located in implementations.py.
    """
    def __init__(self, validation_param, degree=1, do_std=True, do_rm_samples=False,
                 do_rm_features=False, do_tune_hyper=False, do_cross_validate=False):
        self.validation_param = validation_param
        self.degree = degree
        self.do_std = do_std
        self.do_rm_samples = do_rm_samples
        self.do_rm_features = do_rm_features
        self.do_tune_hyper = do_tune_hyper
        self.do_cross_validate = do_cross_validate

    def _run(self, data_y, data_x, data_ids, *hyper):
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
        self.mean, self.std = mean, std

        w_err_hyper_tuples = []  # (w, err) pairs accumulator
        for hyper_params in self._obtain_hyper_params():
            print('Running with hyper parameters:', end=' ')
            print_dict(hyper_params)
            w_err_hyper_tuples.append((self._run(data_y, data_x, data_ids, **hyper_params),
                                        hyper_params))
        
        # Find w that corresponds to minimum error and predict based on that
        (w, err), hyper_params = min(w_err_hyper_tuples, key=lambda x: x[0][1])
        print('Found optimal w with error={err} and hyper parameters:'.format(err=err))
        print_dict(hyper_params)

        self._make_predictions(w)

    @property
    def hyper_params(self):
        raise NotImplementedError

    def _obtain_hyper_params(self):
        if self.do_tune_hyper:
            hyper_providers = {hp: getattr(self, '_tune_{h}'.format(h=hp))()
                               for hp in self.hyper_params.keys()}
            for hyper_values in itertools.product(*hyper_providers.values()):
                yield dict(zip(hyper_providers.keys(), hyper_values))
        else:
            yield self.hyper_params

    @property
    def _trainer(self):
        if self.do_cross_validate:
            return self._train_and_cross_validate
        else:
            return self._train_and_validate

    def _train_and_validate(self, data_y, data_x, data_ids, f, **args):
        ratio = self.validation_param

        # Split data
        train_y, train_x, lc_test_y, lc_test_x = parsers.split_data_rand(data_y, data_x, ratio)
        # call train function
        w, err = f(train_y, train_x, **args)

        # TODO: Check if test error is correct and return it
        test_error = costs.compute_mse(lc_test_y, lc_test_x, w)

        # Predict labels for local testing data
        lc_pred_y = parsers.predict_labels(w, lc_test_x)

        matches = np.sum(lc_test_y == lc_pred_y)
        accuracy = matches / lc_test_y.shape[0]
        print('Accuracy:', accuracy)

        return w, test_error

    def _train_and_cross_validate(self, data_y, data_x, data_ids, f, **args):
        k = self.validation_param
        # Create k subsets of the dataset
        subsets_y, subsets_x = parsers.k_fold_random_split(data_y, data_x, k)

        # Train and validate k times, each time picking subset i as the test set
        avg_test_error = 0
        avg_accuracy = 0

        for i in range(k):
            # Concatenate k-1 subsets 
            train_x = np.concatenate([subsets_x[j] for j in range(k) if j != i], 0)
            train_y = np.concatenate([subsets_y[j] for j in range(k) if j != i], 0)

            # call train function
            w, err = f(train_y, train_x, **args)

            # TODO Return average test error
            # Calculate test error, with subset i as test set
            avg_test_error += costs.compute_mse(subsets_y[i], subsets_x[i], w)

            # Predict labels for local testing data
            lc_pred_y = parsers.predict_labels(w, subsets_x[i])

            matches = np.sum(subsets_y[i] == lc_pred_y)
            avg_accuracy += matches / subsets_y[i].shape[0]

        avg_test_error /= k
        avg_accuracy /= k
        print("Average accuracy:", avg_accuracy)

        return w, avg_test_error

    def _make_predictions(self, w):
        raise NotImplementedError


class GDFitter(Fitter):
    """Fitter implementing linear regression using gradient descent."""

    def __init__(self, validation_param, max_iters, gamma, **kwargs):
        super().__init__(validation_param, **kwargs)
        self.max_iters = max_iters
        self.gamma = gamma

    def _run(self, data_y, data_x, data_ids, **hyper):
        w_init = np.zeros((data_x.shape[1], ))
        f = impl.least_squares_GD
        args = {
            'initial_w': w_init,
            'max_iters': self.max_iters,
            **hyper
        }

        return self._trainer(data_y, data_x, data_ids, f, **args)

    @property
    def hyper_params(self):
        return {'gamma': self.gamma}

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


class SGDFitter(GDFitter):

    def _run(self, data_y, data_x, data_ids, **hyper):
        w_init = np.zeros((data_x.shape[1], ))
        f = impl.least_squares_SGD
        args = {
            'initial_w': w_init,
            'max_iters': self.max_iters,
            **hyper
        }

        return self._trainer(data_y, data_x, data_ids, f, **args)


class RidgeFitter(Fitter):
    """Fitter implementing ridge regression using least squares."""

    def __init__(self, validation_param, lambda_, **kwargs):
        super().__init__(validation_param, **kwargs)
        self.lambda_ = lambda_

    def _run(self, data_y, data_x, data_ids, **hyper):
        f = impl.ridge_regression
        args = {
            **hyper
        }

        return self._trainer(data_y, data_x, data_ids, f, **args)

    @property
    def hyper_params(self):
        return {'lambda_': self.lambda_}

    def _tune_lambda_(self):
        for v in np.logspace(-12, -1, 20):
            yield v
    
    def _make_predictions(self, w):
        test_path = os.path.join('..', 'data', 'test.csv')
        _, test_tx, test_ids = loaders.load_csv_data(test_path)

        if self.do_std:
            test_tx, _, _ = parsers.standardize(test_tx)

        test_tx = parsers.cut_features(test_tx) if self.do_rm_features else test_tx
        pred_y = parsers.predict_labels(w, parsers.build_poly(test_tx, self.degree, True))

        loaders.create_csv_submission(test_ids, pred_y, "ridge.csv")


class LogisticFitter(Fitter):

    def __init__(self, validation_param, max_iters, gamma, **kwargs):
        super().__init__(validation_param, **kwargs)
        self.max_iters = max_iters
        self.gamma = gamma

    def _run(self, data_y, data_x, data_ids, **hyper):
        N, D = data_x.shape
        # add constant term to input samples
        data_x = np.concatenate((np.ones((N, 1)), data_x), axis=1)

        f = impl.logistic_regression
        args = {
            'initial_w': np.zeros((D + 1, )),
            'max_iters': self.max_iters,
            **hyper,
        }

        return self._trainer(data_y, data_x, data_ids, f, **args)

    @property
    def hyper_params(self):
        return {'gamma': self.gamma}
    
    def _tune_gamma(self):
        for v in range(1, 20):
            yield v / 10

    def _make_predictions(self, w):
        test_path = os.path.join('..', 'data', 'test.csv')
        _, test_tx, test_ids = loaders.load_csv_data(test_path)

        if self.do_std:
            test_tx, _, _ = parsers.standardize(test_tx)

        test_tx = parsers.cut_features(test_tx) if self.do_rm_features else test_tx
        pred_y = parsers.predict_labels(w, parsers.build_poly(test_tx, self.degree, True),
                                        is_logistic=True)

        loaders.create_csv_submission(test_ids, pred_y, "logistic.csv")


class RegLogisticFitter(LogisticFitter):

    def __init__(self, validation_param, max_iters, gamma, lambda_, **kwargs):
        super().__init__(validation_param, max_iters, gamma, **kwargs)
        self.lambda_ = lambda_

    def _run(self, data_y, data_x, data_ids, **hyper):
        pass

    @property
    def hyper_params(self):
        return {
            'lambda_': self.lambda_,
            **super().hyper_params,
        }

    def _tune_lambda_(self):
        for v in np.logspace(-12, -1, 20):
            yield v
