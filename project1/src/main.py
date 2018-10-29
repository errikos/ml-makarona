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


if __name__ == '__main__':
    cli(obj={})
