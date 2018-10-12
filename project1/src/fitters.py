# -*- coding: utf-8 -*-
import abc
from util import loaders, parsers, testers


class Fitter(mataclass=abc.ABCMeta):

    def __init__(self, ratio, degree=1, do_std=True, do_rm_samples=False, do_rm_features=False):
        self.ratio = ratio
        self.degree = degree
        self.do_std = do_std
        self.do_rm_samples = do_rm_samples
        self.do_rm_features = do_rm_features
        
    def run(data_x, data_y, data_ids):
        if do_rmv_feat:
            data_x = parsers.cut_features(data_x)
        if do_rmv_samples:
            data_y, data_x = parsers.cut_samples(data_y, data_x)
        if do_standardise:
            data_x, mean, std = parsers.standardize(tmp_tx)
        data_x = parsers.build_poly(data_x, self.degree, True)
        self.data_x, self.data_y, self.data_ids = data_x, data_y, data_ids 
        self.mean, self.std = mean, std


class GDfitter(Fitter):

    def __init__(self, max_iter, gamma, **kwargs):
        super().__init__(ratio, degree, **kwargs)
        self.max_iter = max_iter
        self.gamma = gamma

    def run(data):
        super().run()
        train_x, train_y, train_ids, lc_test_x, lc_test_y, lc_test_ids = \
            parsers.split_data_rand(self.data_x, self.data_y, self.data_ids, self.ratio)
