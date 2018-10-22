#!/usr/bin/env python3
import click
import fitters as ft
from util import loaders
import os


@click.group()
def cli():
    pass


@cli.command(help='Gradient Descent')
def gd():

    fitter = ft.GDFitter(0.8, 4000, 0.0001, degree=3, do_std = True, do_rm_samples = False)
    data_path = os.path.join("..", "data", "train.csv")
    tmp_y, tmp_tx, tmp_ids = loaders.load_csv_data(data_path)
    fitter.run(tmp_y, tmp_tx, tmp_ids)


@cli.command(help='Stochastic Gradient Descent')
def sgd():
    pass


@cli.command(help='Least Squares')
def least():
    pass


@cli.command(help='Ridge Regression')
def ridge():
    fitter = ft.RidgeFitter(0.8, 0.01, degree=1, do_std = True)
    data_path = os.path.join("..", "data", "train.csv")
    tmp_y, tmp_tx, tmp_ids = loaders.load_csv_data(data_path)
    fitter.run(tmp_y, tmp_tx, tmp_ids)


@cli.command(help='Logistic Regression')
def log():
    pass


@cli.command(help='Regularised Logistic Regression')
def reglog():
    pass


if __name__ == '__main__':
    cli()
