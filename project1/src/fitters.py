# -*- coding: utf-8 -*-
import abc
import implementations as impl
from util import loaders, parsers, testers


class Fitter(metaclass=abc.ABCMeta):

    def __init__(self, ratio, degree=1, do_std=True, do_rm_samples=False, do_rm_features=False):
        self.ratio = ratio
        self.degree = degree
        self.do_std = do_std
        self.do_rm_samples = do_rm_samples
        self.do_rm_features = do_rm_features
        
    def run(self, data_x, data_y, data_ids):
        if self.do_rm_features:
            data_x = parsers.cut_features(data_x)
        if self.do_rm_samples:
            data_y, data_x = parsers.cut_samples(data_y, data_x)
        # if self.do_std:
            # for i  in range(x.shape[1]): IndexError: tuple index out of range
            # data_x, mean, std = parsers.standardize(data_x)
        data_x = parsers.build_poly(data_x, self.degree, True)
        self.data_x, self.data_y, self.data_ids = data_x, data_y, data_ids 
        self.mean, self.std = mean, std


class GD_fitter(Fitter):

    def __init__(self, ratio, max_iter, gamma, **kwargs):
        super().__init__(ratio, **kwargs)
        self.max_iter = max_iter
        self.gamma = gamma

    def run(self, data_x, data_y, data_ids):
        super().run(data_x, data_y, data_ids)
        # Split data to training and testing
        train_x, train_y, train_ids, lc_test_x, lc_test_y, lc_test_ids = \
            parsers.split_data_rand(self.data_x, self.data_y, self.data_ids, self.ratio)
        # Train the model
        w_init = np.zeros((train_x.shape[1], 1))
        w, mse = impl.least_squares_GD(train_y, train_x, w_init, self.max_iter, self.gamma)
        # Do testing ...
        # Print test results...
        # Create predictions for test.csv...