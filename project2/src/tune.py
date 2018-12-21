#!/usr/bin/env python3
import click
import itertools

import helpers

import surprise
import numpy as np


def print_dict(d, delim='=', prefix=''):
    print(prefix)
    if not d:
        print('(none)')
    for k, v in d.items():
        print('{k}{d}{v}'.format(k=k, d=delim, v=v), end=' ')
    print()


class Tuner(object):
    """Generic tuner class implementing grid search for the given parameters and ranges."""

    def __init__(self, input_path, **params):
        """Initialize a Tuner object.

        :param params: The parameters to tune. A dict of <name: str, range> pairs.
        """
        self.input_path = input_path
        self.params = {name: range_ for name, range_ in params.items() if range_ is not None}
        self.tuned = False
        self.optimal = {}

    def _print_params(self):
        print('Will tune based on the following parameter values:')
        for param, vals in self.params.items():
            print('  - {param}: {vals}'.format(param=param, vals=','.join(map(str, vals))))

    def tune(self, load_fn, split_fn, train_fn, eval_fn, ratio, **options):
        """Given the functions indicated below, discover the optimal values for the parameters to tune.

        :param load_fn: A function that, given the input path, returns a dataset object.
        :param split_fn: A function that, given the dataset object, splits it and returns the training/testing datasets.
        :param train_fn: A function that, given the training dataset, returns the trained model object.
        :param eval_fn: A function that, given the model and the testing dataset, returns the evaluated RMSE.
        :param ratio: The ratio for the training/testing split (a float in the range (0, 1)).
        :param options: Any additional options (key-word arguments) to pass to the training function.
        """
        print('Tuning', train_fn.__name__)
        self._print_params()

        ratings = load_fn(self.input_path)
        training, testing = split_fn(ratings, ratio)

        results = []
        for param_values in itertools.product(*self.params.values()):
            # equivalent to nested for-loops for all parameter ranges
            selected_params = dict(zip(self.params.keys(), param_values))
            print_dict(selected_params, prefix='Running with: ')
            model = train_fn(training, **{**selected_params, **options})
            rmse = eval_fn(model, testing)
            print('RMSE:', rmse)
            results.append((selected_params, rmse))

        best_params, best_rmse = min(results, key=lambda t: t[1])
        print('RMSE:', best_rmse)
        print('Results:')
        for param, val in best_params.items():
            print('  - {param}: {val}'.format(param=param, val=val))

    def get_optimal(self):
        if not self.tuned:
            raise RuntimeError('Cannot obtain optimal values before tuning.')


class SurpriseTuner(Tuner):
    """Tuner class for surprise-based models (uses surprise.model_selection.search.GridSearchCV)."""

    def tune(self, train_class, **options):
        """Discover the optimal values for the parameters to tune, using the indicated surprise library train class.
        Uses the surprise library grid search cross-validate (surprise.model_selection.GridSearchCV) class.

        :param train_class: An instance of the surprise.prediction_algorithms.algo_base.AlgoBase class.
        :param options: Any additional options (key-word arguments) to be passed to the grid search algorithm.
        :return:
        """
        self._print_params()

        df = helpers.read_to_df(self.input_path)
        ratings = surprise.Dataset.load_from_df(df, surprise.Reader())

        grid_search = surprise.model_selection.search.GridSearchCV(
            algo_class=train_class,
            param_grid={**self.params, **options},
            measures=['rmse'],
            n_jobs=-1,  # enable parallel execution
        )
        grid_search.fit(ratings)

        print('RMSE:', grid_search.best_score['rmse'])
        print('Results:')
        for param, val in grid_search.best_params['rmse'].items():
            print('  - {param}: {val}'.format(param=param, val=val))


class TuneParamType(click.ParamType):
    name = 'START:END:STEP'

    def __init__(self, type_):
        self.type_ = type_

    def convert(self, value, param, ctx):
        if not value:
            return value
        start, end, step = map(self.type_, value.split(':'))
        return np.arange(start, end+step, step)


@click.group(help='Tuners for various recommendation algorithms.')
@click.pass_context
def cli(ctx, **kwargs):
    ctx.obj.update(**kwargs)


@cli.command(help='Alternating Least Squares (ALS) tuning.')
@click.argument('input_path', type=click.Path(exists=True))
@click.option('-e', '--epochs', 'epochs', type=TuneParamType(int), help='Tune the number of epochs.')
@click.option('-l', '--lambda', 'lambda_', type=TuneParamType(float), help='Tune the λ parameter.')
@click.option('-r', '--rank', type=TuneParamType(int), help='Tune the rank (number of latent features).')
@click.option('--ratio', metavar='RATIO', type=float, default=0.9,
              help='The training ratio for the train/test split (default: 0.9).')
def als(input_path, ratio, **params):
    import models.als_spark as als_spark
    tuner = Tuner(input_path, **params)
    tuner.tune(als_spark.load_ratings, als_spark.split_train_test, als_spark.train, als_spark.evaluate, ratio)


@cli.command(help='Co-clustering tuning.')
@click.argument('input_path', type=click.Path(exists=True))
@click.option('-e', '--epochs', 'n_epochs', type=TuneParamType(int), help='Tune the number of epochs.')
@click.option('-uc', '--user-clusters', 'n_cltr_u', type=TuneParamType(int), help='Tune the number of user clusters.')
@click.option('-ic', '--item-clusters', 'n_cltr_i', type=TuneParamType(int), help='Tune the number of item clusters.')
def co_cluster(input_path, **params):
    tuner = SurpriseTuner(input_path, **params)
    tuner.tune(surprise.CoClustering)


@cli.command(help='KNN item-based tuning.')
@click.argument('input_path', type=click.Path(exists=True))
@click.option('-b', '--with-baseline', is_flag=True, help='Enable baseline.')
@click.option('-k', '--neighbours', 'k', type=TuneParamType(int), help='Tune the number of neighbours.')
def item_based(input_path, with_baseline, **params):
    tuner = SurpriseTuner(input_path, **params)
    tuner.tune(**{
        'train_class': surprise.KNNWithMeans if not with_baseline else surprise.KNNBaseline,
        'name': ['pearson'] if not with_baseline else ['pearson_baseline'],
        'user_based': [False],
    })


@cli.command(help='SVD tuning.')
@click.argument('input_path', type=click.Path(exists=True))
@click.option('-e', '--epochs', 'n_epochs', type=TuneParamType(int), help='Tune the number of epochs.')
@click.option('-f', '--factors', 'n_factors', type=TuneParamType(int), help='Tune the number of factors.')
@click.option('-l', '--learn-rate', 'lr_all', type=TuneParamType(float), help='Tune the learning rate.')
@click.option('-r', '--reg-term', 'reg_all', type=TuneParamType(float), help='Tune the regularization term.')
def svd(input_path, **params):
    tuner = SurpriseTuner(input_path, **params)
    tuner.tune(surprise.SVD)


@cli.command(name='svd++', help='SVD++ tuning.')
@click.argument('input_path', type=click.Path(exists=True))
@click.option('-e', '--epochs', 'n_epochs', type=TuneParamType(int), help='Tune the number of epochs.')
@click.option('-f', '--factors', 'n_factors', type=TuneParamType(int), help='Tune the number of factors.')
@click.option('-l', '--learn-rate', 'lr_all', type=TuneParamType(float), help='Tune the learning rate.')
@click.option('-r', '--reg-term', 'reg_all', type=TuneParamType(float), help='Tune the regularization term.')
def svdpp(input_path, **params):
    tuner = SurpriseTuner(input_path, **params)
    tuner.tune(surprise.SVDpp)


@cli.command(help='KNN user-based tuning.')
@click.argument('input_path', type=click.Path(exists=True))
@click.option('-b', '--with-baseline', is_flag=True, help='Enable baseline.')
@click.option('-k', '--neighbours', 'k', type=TuneParamType(int), help='Tune the number of neighbours.')
def user_based(input_path, with_baseline, **params):
    tuner = SurpriseTuner(input_path, **params)
    tuner.tune(**{
        'train_class': surprise.KNNWithMeans if not with_baseline else surprise.KNNBaseline,
        'name': ['pearson'] if not with_baseline else ['pearson_baseline'],
        'user_based': [True],
    })


@cli.command(help='Fit the KNN with z-score for each user algorithm.')
@click.argument('input_path', type=click.Path(exists=True))
@click.option('-k', '--neighbours', 'k', type=TuneParamType(int), help='Tune the number of neighbours.')
def z_score(input_path, **params):
    tuner = SurpriseTuner(input_path, **params)
    tuner.tune(surprise.KNNWithZScore)


if __name__ == '__main__':
    cli(obj={})
