# EPFL Machine Learning 2018-2019
## Project 1: Higgs Boson challenge

### Introduction

The project consists of Python implementations of six regression methods,
the core of which lie within the `implementations.py` file:
- Gradient Descent
- Stochastic Gradient Descent
- Least Squares (using normal equations)
- Ridge Regression (using normal equations)
- Logistic Regression (using Gradient Descent of Newton's method)
- Regularised Logistic Regression (using Gradient Descent)

### Class hierarchy

For a nice, object-oriented interface, a hierarchy of classes is also
provided, one for each of the above methods, in `fitters.py`. All fitter classes
inherit from the abstract base class `Fitter`. This layout avoids a substantial
amount of code duplication, as most functions are implemented by the `Fitter`
class and shared among the subclasses.
In addition, some of the subclasses have very similar behaviour
(e.g. Normal/Stochastic Gradient Descent), so the second one can derive from
the first one and only change the implementation to call.

### Utilities

Functions that manipulate data, but do not consist part of the behaviour
of a `Fitter` are placed outside the hierarchy into the `util` package.
There are two such groups of functions:
- loaders: load data from CSV and create the submission CSV
- modifiers: modify or augment the dataset and/or its features

Other function groups that deserved their own packages are the following:
- costs: contains cost function implementations
- gradients: contains implementations of gradient calculators

### Dependencies

You should have no trouble running the program, as long as you are using
an Anaconda3-based Python distribution. The only dependencies are `numpy`
and `click` (for the command-line interface).

### Running the Fitters

There are two ways to run the fitters:
- To exactly reproduce the Kaggle score, run `run.py`.
- To fine tune and run a fitter, we have created a
    [click](https://click.palletsprojects.com/en/7.x/)-based command line interface,
    which resides in `main.py`. Just run `python3 main.py --help` to get an overview
    of the accepted options.