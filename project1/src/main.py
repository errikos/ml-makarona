#!/usr/bin/env python3
import click
import os

import numpy as np

import fitters as ft
from util import loaders


DEFAULT_DATA_PATH = os.path.join('..', 'data')


def _res_validation_param(param, do_cross_validation):
    if not param:
        return param
    if do_cross_validation:
        param = int(param)
        return param if 1 < param else None
    else:
        param = float(param)
        return param if 0.0 < param <= 1.0 else None


def _load_data(data_path, is_logistic=False):
    train_y, train_tx, train_ids = loaders.load_csv_data(os.path.join(data_path, 'train.csv'),
                                                         is_logistic=is_logistic)
    _, test_tx, test_ids = loaders.load_csv_data(os.path.join(data_path, 'test.csv'),
                                                 is_logistic=is_logistic)
    return train_y, train_tx, train_ids, test_tx, test_ids


"""
The command line interface options are shared among the various methods and must be provided
before the method is specified. Afterwards, the method arguments must be specified in order.

Some execution examples:
    $ python main.py --validate 0.8 gd 4000 0.001
    $ python main.py --cross --validate 4 gd 4000 0.001
    $ python main.py --validate 0.9 --degree 5 gd 4000 0.001
    $ python main.py --validate 0.9 --degree 5 reglog 4000 0.001 0.05
"""

@click.group()
@click.option('--std', is_flag=True,
              help='Standardize data.')
@click.option('--tune', is_flag=True,
              help='Tune hyper parameters.')
@click.option('--cross', is_flag=True,
              help='Do cross validation.')
@click.option('--rm-samples', is_flag=True,
              help='Remove samples having at least one -999 value.')
@click.option('--rm-features', is_flag=True,
              help='Remove features having at least one -999 value.')
@click.option('--data-path', metavar='PATH',
              help='Directory containing train.csv and test.csv.')
@click.option('--validate', metavar='RATIO_OR_K',
              help='Validation method parameter (split ratio or k for k-fold cross validation).')
@click.option('--degree', metavar='DEGREE', type=int, default=1,
              help='Polynomial degree for enhanced feature vectors.')
@click.pass_context
def cli(ctx, **kwargs):
    ctx.ensure_object(dict)
    ctx.obj.update(**kwargs)
    # resolve data path
    ctx.obj.update(data_path=kwargs['data_path'] or DEFAULT_DATA_PATH)
    # ensure proper validation parameter
    ctx.obj.update(validation_param=_res_validation_param(kwargs['validate'], kwargs['cross']))
    if not ctx.obj['validation_param']:
        raise click.ClickException('missing or invalid value for validation parameter (ratio or k)')


@cli.command(help='Gradient Descent')
@click.argument('max_iters', type=int)
@click.argument('gamma', type=float)
@click.pass_context
def gd(ctx, max_iters, gamma):
    fitter = ft.GDFitter(max_iters, gamma, **ctx.obj)  # create fitter object
    fitter.run(*_load_data(ctx.obj['data_path']))  # load data and run


@cli.command(help='Stochastic Gradient Descent')
@click.argument('max_iters', type=int)
@click.argument('gamma', type=float)
@click.pass_context
def sgd(ctx, max_iters, gamma):
    fitter = ft.SGDFitter(max_iters, gamma, **ctx.obj)  # create fitter object
    fitter.run(*_load_data(ctx.obj['data_path']))  # load data and run


@cli.command(help='Least Squares')
@click.pass_context
def least(ctx):
    fitter = ft.LeastFitter(**ctx.obj)  # create fitter object
    fitter.run(*_load_data(ctx.obj['data_path']))  # load data and run


@cli.command(help='Ridge Regression')
@click.argument('lambda_', type=float, metavar='LAMBDA')
@click.pass_context
def ridge(ctx, lambda_):
    fitter = ft.RidgeFitter(lambda_, **ctx.obj)  # create fitter object
    fitter.run(*_load_data(ctx.obj['data_path']))  # load data and run


@cli.command(help='Logistic Regression')
@click.argument('max_iters', type=int)
@click.argument('gamma', type=float)
@click.option('--newton', is_flag=True, help="Run Newton's method.")
@click.pass_context
def log(ctx, max_iters, gamma, newton=False):
    fitter = ft.LogisticFitter(max_iters, gamma, newton, **ctx.obj)  # create fitter object
    fitter.run(*_load_data(ctx.obj['data_path'], is_logistic=True))  # load data and run


@cli.command(help='Regularised Logistic Regression')
@click.argument('max_iters', type=int)
@click.argument('gamma', type=float)
@click.argument('lambda_', type=float, metavar='LAMBDA')
@click.option('--newton', is_flag=True, help="Run Newton's method.")
@click.pass_context
def reglog(ctx, max_iters, gamma, lambda_, newton):
    fitter = ft.RegLogisticFitter(max_iters, gamma, lambda_, newton, **ctx.obj)  # create fitter
    fitter.run(*_load_data(ctx.obj['data_path'], is_logistic=True))  # load data and run


def smart_training():
    from util import parsers
    from implementations import ridge_regression

    train_y, train_tx, train_ids, test_tx, test_ids = _load_data(DEFAULT_DATA_PATH)

    feat = train_tx[:, 0]
    invalid = feat == -999
    valid = feat != -999
    median = np.median(feat[valid])
    feat[invalid] = median
    train_tx[:, 0] = feat

    # split train data
    jet_feat = 22
    train_jet0_tx = train_tx[train_tx[:, jet_feat] == 0]
    train_jet0_tx = np.delete(train_jet0_tx, [22, 4, 5, 6, 12, 23, 24, 25, 26, 27, 28, 29], axis=1)
    train_jet0_y = train_y[train_tx[:, jet_feat] == 0]

    train_jet1_tx = train_tx[train_tx[:, jet_feat] == 1]
    train_jet1_tx = np.delete(train_jet1_tx, [22, 4, 5, 6, 12, 26, 27, 28], axis=1)
    train_jet1_y = train_y[train_tx[:, jet_feat] == 1]

    train_jetR_tx = train_tx[train_tx[:, jet_feat] >= 2]
    train_jetR_tx = np.delete(train_jetR_tx, [22], axis=1)
    train_jetR_y = train_y[train_tx[:, jet_feat] >= 2]

    # split test data
    # TODO

    # build polynomials
    degree = 8
    train_jet0_tx = parsers.build_poly(train_jet0_tx, degree=4)
    train_jet1_tx = parsers.build_poly(train_jet1_tx, degree=9)
    train_jetR_tx = parsers.build_poly(train_jetR_tx, degree=10)

    # split data to train and test
    ratio = 0.8
    seed = 155
    train_jet0_y, train_jet0_tx, test_jet0_y, test_jet0_tx = parsers.split_data_rand(
        train_jet0_y, train_jet0_tx, ratio, seed)
    train_jet1_y, train_jet1_tx, test_jet1_y, test_jet1_tx = parsers.split_data_rand(
        train_jet1_y, train_jet1_tx, ratio, seed)
    train_jetR_y, train_jetR_tx, test_jetR_y, test_jetR_tx = parsers.split_data_rand(
        train_jetR_y, train_jetR_tx, ratio, seed)

    print(train_jet0_tx.shape)
    print(train_jet1_tx.shape)
    print(train_jetR_tx.shape)

    # train three models
    lambda1_ = 0.01
    lambda2_ = 0.01
    lambdaR_ = 0.01
    w0, loss0 = ridge_regression(train_jet0_y, train_jet0_tx, lambda_)
    w1, loss1 = ridge_regression(train_jet1_y, train_jet1_tx, lambda_)
    wR, lossR = ridge_regression(train_jetR_y, train_jetR_tx, lambda_)

    # get local testing accuracy
    train_y_preds = parsers.predict_labels(w0, train_jet0_tx)

    tr_acc = []
    te_acc = []
    for w, tr_y, tr_x, te_y, te_x in zip([w0, w1, wR],
                                         [train_jet0_y, train_jet1_y, train_jetR_y],
                                         [train_jet0_tx, train_jet1_tx, train_jetR_tx],
                                         [test_jet0_y, test_jet1_y, test_jetR_y],
                                         [test_jet0_tx, test_jet1_tx, test_jetR_tx]):
        train_y_preds = parsers.predict_labels(w, tr_x)
        matches_tr = np.sum(tr_y == train_y_preds)
        accuracy_tr = matches_tr / tr_y.shape[0]
        tr_acc.append(accuracy_tr)
        print('Training Accuracy:', accuracy_tr)

        test_y_preds = parsers.predict_labels(w, te_x)
        matches_te = np.sum(te_y == test_y_preds)
        accuracy_te = matches_te / te_y.shape[0]
        te_acc.append(accuracy_te)
        print('Testing Accuracy:', accuracy_te)


    print('-------------------------------------------------------------')
    print('Weighted average TR:', (len(train_jet0_tx) / 199999 * tr_acc[0]) + 
                                  (len(train_jet1_tx) / 199999 * tr_acc[1]) +
                                  (len(train_jetR_tx) / 199999 * tr_acc[2]))
    print('Weighted average TE:', (len(test_jet0_tx) / 50001 * te_acc[0]) + 
                                  (len(test_jet1_tx) / 50001 * te_acc[1]) +
                                  (len(test_jetR_tx) / 50001 * te_acc[2]))

    

if __name__ == '__main__':
    # cli(obj={})
    smart_training()
