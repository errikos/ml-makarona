#!/usr/bin/env python3
import click
import fitters as ft


@click.group()
def cli():
    pass


@cli.command(help='Gradient Descent')
def gd():
    fitter = ft.GD_fitter(0.8,1000,0.5)


@cli.command(help='Stochastic Gradient Descent')
def sgd():
    pass


@cli.command(help='Least Squares')
def least():
    pass


@cli.command(help='Ridge Regression')
def ridge():
    pass


@cli.command(help='Logistic Regression')
def log():
    pass


@cli.command(help='Regularised Logistic Regression')
def reglog():
    pass


if __name__ == '__main__':
    cli()
