# -*- coding: utf-8 -*-
import abc
import numpy as np
import itertools
import os

import implementations as impl
import costs
from util import loaders, parsers
import costs


def print_dict(d, delim='='):
    if not d:
        print('(none)', end='')
    for k, v in d.items():
        print('{k}{d}{v}'.format(k=k, d=delim, v=v), end=' ')


class Fitter(metaclass=abc.ABCMeta):
    """Base class for Fitter objects.

    Each fitter encapsulates a fitter behaviour and provides a nice interface
    for settings parameters, tuning hyper-parameters, etc.

    The core method implementations are located in implementations.py.
    """
    def __init__(self, validation_param, degree=1, std=True, eliminate_minus_999=False,
                 drop_minus_999_features=False, tune=False, cross=False, **kwargs):
        self.validation_param = validation_param
        self.degree = degree
        self.do_std = std
        self.do_eliminate_minus_999 = eliminate_minus_999
        self.do_drop_minus_999_features = drop_minus_999_features
        self.do_tune_hyper = tune
        self.do_cross_validate = cross

    def _run(self, data_y, data_x, data_ids, initial_w=None, **hyper):
        raise NotImplementedError

    def run(self, data_y, data_x, data_ids, test_x, test_ids):
        if self.do_drop_minus_999_features:
            print('Dropping features containing at least one -999 value...', end=' ', flush=True)
            data_x = parsers.drop_minus_999_features(data_x)
            print('DONE')
        if self.do_eliminate_minus_999:
            print('Eliminating -999 values by setting them to feature median...', end=' ', flush=True)
            data_x = parsers.eliminate_minus_999(data_x)
            print('DONE')

        # Build polynomial
        data_x = parsers.build_poly(data_x, self.degree, True)

        if self.do_std:
            print('Standardising...', end=' ', flush=True)
            data_x = parsers.standardize(data_x)
            print('DONE')

        # Find a good initial w
        initial_w, _ = impl.ridge_regression(data_y, data_x, lambda_=0.1)

        w_err_hyper_tuples = []  # (w, err, acc) triplets accumulator
        for hyper_params in self._obtain_hyper_params():
            print('Running with hyper parameters:', end=' ')
            print_dict(hyper_params)
            print()

            result = self._run(data_y, data_x, data_ids, initial_w, **hyper_params)
            w_err_hyper_tuples.append((result, hyper_params))
        
        # Find w that corresponds to minimum error and predict based on that
        (w, err, acc), hyper_params = min(w_err_hyper_tuples, key=lambda x: x[0][1])
        print('Found optimal w with error={err}, accuracy={acc}'.format(err=err, acc=acc),
              'and hyper parameters:', end=' ')
        print_dict(hyper_params)
        print()

        if np.isnan(err):
            print('Error is infinite, computation has probably diverged.',
                  'Abandoning predictions!')
            return

        self._make_predictions(w, test_x, test_ids)

    @property
    def hyper_params(self):
        return {}

    def penalization(self, w):
        return 0.0

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

    def _train_and_validate(self, data_y, data_x, data_ids, train_f, cost_f,
                            is_logistic=False, **train_args):
        ratio = self.validation_param

        # Split data
        train_y, train_x, lc_test_y, lc_test_x = parsers.split_data_rand(data_y, data_x, ratio)
        # call train function
        w, err = train_f(train_y, train_x, **train_args)

        test_error = cost_f(lc_test_y, lc_test_x, w)
        test_error += self.penalization(w)

        # Predict labels for local training data
        lc_pred_y_tr = parsers.predict_labels(w, train_x, is_logistic=is_logistic)
        # Predict labels for local testing data
        lc_pred_y = parsers.predict_labels(w, lc_test_x, is_logistic=is_logistic)

        # Training matches
        matches_tr = np.sum(train_y == lc_pred_y_tr)
        accuracy_tr = matches_tr / train_y.shape[0]
        # Local Testing matches
        matches = np.sum(lc_test_y == lc_pred_y)
        accuracy = matches / lc_test_y.shape[0]
        print('Train-Validate: Training error={err}, Training accuracy={acc}'.format(err=err, acc=accuracy_tr))
        print('Train-Validate: Test error={err}, Test accuracy={acc}'.format(err=test_error, acc=accuracy))

        return w, test_error, accuracy

    def _train_and_cross_validate(self, data_y, data_x, data_ids, train_f, cost_f,
                                  is_logistic=False, **train_args):
        k = self.validation_param
        # Create k subsets of the dataset
        subsets_y, subsets_x = parsers.k_fold_random_split(data_y, data_x, k)

        # Train and validate k times, each time picking subset i as the test set
        avg_test_error = 0
        avg_test_accuracy = 0
        avg_train_accuracy = 0

        for i in range(k):
            # Concatenate k-1 subsets 
            train_x = np.concatenate([subsets_x[j] for j in range(k) if j != i], 0)
            train_y = np.concatenate([subsets_y[j] for j in range(k) if j != i], 0)

            # call train function
            w, _ = train_f(train_y, train_x, **train_args)

            # Calculate test error, with subset i as test set
            avg_test_error += cost_f(subsets_y[i], subsets_x[i], w)
            avg_test_error += self.penalization(w)

            # Predict labels for local training data
            lc_pred_y_tr = parsers.predict_labels(w, train_x, is_logistic=is_logistic)
            matches_tr = np.sum(train_y == lc_pred_y_tr)
            avg_train_accuracy += matches_tr / train_y.shape[0]

            # Predict labels for local testing data
            lc_pred_y = parsers.predict_labels(w, subsets_x[i], is_logistic=is_logistic)
            matches = np.sum(subsets_y[i] == lc_pred_y)
            avg_test_accuracy += matches / subsets_y[i].shape[0]

        avg_test_error /= k
        avg_test_accuracy /= k
        avg_train_accuracy /= k

        print('Train-CrossValidate: Train: AVG accuracy={acc}'.format(acc=avg_train_accuracy))
        print('Train-CrossValidate: Test: AVG error={err}, AVG accuracy={acc}'.format(err=avg_test_error,
                                                                                      acc=avg_test_accuracy))

        return w, avg_test_error, avg_test_accuracy

    def _make_predictions(self, w, test_tx, test_ids):
        raise NotImplementedError

    def _make_predictions_core(self, w, test_tx, test_ids, filename, **kwargs):
        if self.do_drop_minus_999_features:
            test_tx = parsers.drop_minus_999_features(test_tx)
        if self.do_eliminate_minus_999:
            test_tx = parsers.eliminate_minus_999(test_tx)
        
        # augment features
        test_tx = parsers.build_poly(test_tx, self.degree, True)

        if self.do_std:
            test_tx = parsers.standardize(test_tx)

        test_tx = parsers.drop_minus_999_features(test_tx) if self.do_drop_minus_999_features else test_tx
        pred_y = parsers.predict_labels(w, test_tx, **kwargs)

        loaders.create_csv_submission(test_ids, pred_y, filename)


class GDFitter(Fitter):
    """Fitter implementing linear regression using gradient descent."""

    def __init__(self, max_iters, gamma, **kwargs):
        super().__init__(**kwargs)
        self.max_iters = max_iters
        self.gamma = gamma

    def _run(self, data_y, data_x, data_ids, initial_w=None, **hyper):
        _, D = data_x.shape

        f = impl.least_squares_GD
        args = {
            'initial_w': initial_w if initial_w is not None else np.zeros((D, )),
            'max_iters': self.max_iters,
            **hyper
        }

        return self._trainer(data_y, data_x, data_ids, train_f=f, cost_f=costs.compute_mse, **args)

    @property
    def hyper_params(self):
        return {'gamma': self.gamma}

    def _tune_gamma(self):
        for v in range(1, 20):
            yield v / 10
    
    def _make_predictions(self, w, test_tx, test_ids):
        self._make_predictions_core(w, test_tx, test_ids, 'gd.csv')


class SGDFitter(GDFitter):
    """Fitter implementing linear regression using stochastic gradient descent."""

    def _run(self, data_y, data_x, data_ids, initial_w=None, **hyper):
        _, D = data_x.shape

        f = impl.least_squares_SGD
        args = {
            'initial_w': initial_w if initial_w is not None else np.zeros((D, )),
            'max_iters': self.max_iters,
            **hyper
        }

        return self._trainer(data_y, data_x, data_ids, train_f=f, cost_f=costs.compute_mse, **args)


class LeastFitter(Fitter):
    """Fitter implementing least squares using normal equations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self, data_y, data_x, data_ids, initial_w=None, **hyper):
        f = impl.least_squares
        args = {
            **hyper
        }

        return self._trainer(data_y, data_x, data_ids, train_f=f, cost_f=costs.compute_mse, **args)

    def _make_predictions(self, w, test_tx, test_ids):
        self._make_predictions_core(w, test_tx, test_ids, 'least.csv')


class RidgeFitter(Fitter):
    """Fitter implementing ridge regression using least squares."""

    def __init__(self, lambda_, **kwargs):
        super().__init__(**kwargs)
        self.lambda_ = lambda_

    def _run(self, data_y, data_x, data_ids, initial_w=None, **hyper):
        f = impl.ridge_regression
        args = {
            **hyper
        }

        return self._trainer(data_y, data_x, data_ids, train_f=f, cost_f=costs.compute_rmse, **args)

    @property
    def hyper_params(self):
        return {'lambda_': self.lambda_}

    def _tune_lambda_(self):
        for v in np.logspace(-12, -1, 20):
            yield v
    
    def _make_predictions(self, w, test_tx, test_ids):
        self._make_predictions_core(w, test_tx, test_ids, 'ridge.csv')


class LogisticFitter(Fitter):
    """Fitter implementing logistic regression using Newton's method."""

    def __init__(self, max_iters, gamma, newton=False, **kwargs):
        super().__init__(**kwargs)
        self.max_iters = max_iters
        self.gamma = gamma
        self.newton = newton

    def _run(self, data_y, data_x, data_ids, initial_w=None, **hyper):
        _, D = data_x.shape

        f = impl.logistic_regression
        args = {
            'initial_w': initial_w if initial_w is not None else np.zeros((D, )),
            'max_iters': self.max_iters,
            'newton': self.newton,
            **hyper,
        }

        return self._trainer(data_y, data_x, data_ids, train_f=f,
                             cost_f=costs.compute_log_likelihood_error, is_logistic=True, **args)

    @property
    def hyper_params(self):
        return {'gamma': self.gamma}
    
    def _tune_gamma(self):
        for v in range(1, 20):
            yield v / 100

    def _make_predictions(self, w, test_tx, test_ids):
        self._make_predictions_core(w, test_tx, test_ids, 'logistic.csv', is_logistic=True)


class RegLogisticFitter(LogisticFitter):
    """Fitter implementing regularised logistic regression using Newton's method."""

    def __init__(self, max_iters, gamma, lambda_, newton, **kwargs):
        super().__init__(max_iters, gamma, newton, **kwargs)
        self.lambda_ = lambda_

    def _run(self, data_y, data_x, data_ids, initial_w=None, **hyper):
        _, D = data_x.shape

        f = impl.reg_logistic_regression
        args = {
            'initial_w': initial_w if initial_w is not None else np.zeros((D, )),
            'max_iters': self.max_iters,
            'newton': self.newton,
            **hyper,
        }

        return self._trainer(data_y, data_x, data_ids, train_f=f,
                             cost_f=costs.compute_log_likelihood_error, is_logistic=True, **args)

    @property
    def hyper_params(self):
        return {
            'lambda_': self.lambda_,
            **super().hyper_params,
        }

    def _tune_lambda_(self):
        for v in range(1, 11):
            yield v * 1000

    def penalization(self, w):
        lambda_ = self.lambda_
        return (lambda_ / 2.0) * w.dot(w)