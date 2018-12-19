#!/usr/bin/env python3
import click
import surprise

import helpers


class Model(object):
    """Generic model class for fitting a recommendation model."""

    def __init__(self, **kwargs):
        """Initialize a Model object."""
        self.training_path = kwargs.get('train_data_path')
        self.predict_path = kwargs.get('predict_data_path')
        self.output_path = kwargs.get('output_path')

    def train(self, load_fn, train_fn, predict_fn, store_fn, **options):
        """Given the functions indicated below, load the training and prediction datasets,
        train with the training dataset and predict the values for the prediction dataset.
        Write results in output path.

        :param load_fn: A function that, given the input path, returns a dataset object.
        :param train_fn: A function that, given the training dataset, returns the trained model object.
        :param predict_fn: A function that, given the trained model and the prediction dataset, returns the
                           predicted values for all (user, item) pairs in the prediction dataset.
        :param store_fn: A function that, given the predicted values and the output path, writes the predicted
                         values to that path.
        :param options: Any additional options (key-word arguments) to pass to the training function.
        """
        training_dataset = load_fn(self.training_path)
        prediction_dataset = load_fn(self.predict_path)
        model = train_fn(training_dataset, **options)
        predictions = sorted(predict_fn(model, prediction_dataset), key=lambda t: (t[1], t[0]))
        store_fn(self.output_path, predictions)


class SurpriseModel(Model):
    """Specialised class for surprise-based recommendation models."""

    @staticmethod
    def _load_data(path):
        reader = surprise.Reader(line_format='user item rating', sep=',', skip_lines=1)
        return surprise.Dataset.load_from_file(path, reader)

    def train(self, train_class, **options):
        # load training dataset and train model
        training = SurpriseModel._load_data(self.training_path).build_full_trainset()
        algorithm = train_class(**options)
        algorithm.fit(training)

        def predict(user, item):
            prediction = algorithm.predict(user, item)
            return user, item, helpers.clip(prediction.est)
        # load prediction dataset and make predictions
        to_predict = map(lambda r: r.strip().split(',')[:2], helpers.read_lines(self.predict_path, header=False))
        predictions = map(lambda pair: predict(*pair), to_predict)  # make predictions

        # write to output file
        helpers.write_normalized(self.output_path, predictions)


@click.group(help='Runner for training the recommendation algorithms.')
@click.argument('train_data_path', type=click.Path(exists=True))
@click.argument('predict_data_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path(exists=False))
@click.pass_context
def cli(ctx, **kwargs):
    ctx.obj.update(**kwargs)


@cli.command(help='Fit the Alternating Least Squares algorithm (ALS).')
@click.option('-e', '--epochs', 'epochs', type=int, required=True, help='The number of epochs (iterations).')
@click.option('-l', '--lambda', 'lambda_', type=float, required=True,  help='The λ parameter.')
@click.option('-r', '--rank', type=int, required=True, help='The rank (number of latent features).')
@click.option('-s', '--seed', type=int, required=False, help='The random seed for the ALS algorithm.')
@click.pass_context
def als(ctx, **params):
    import spark.als as als_spark
    model = Model(**ctx.obj)
    model.train(als_spark.load_ratings, als_spark.train, als_spark.predict, helpers.write_normalized, **params)


@cli.command(help='Fit the Co-Clustering algorithm.')
@click.option('-e', '--epochs', 'n_epochs', type=int, required=True, help='The number of epochs.')
@click.option('-uc', '--user-clusters', 'n_cltr_u', type=int, required=True, help='The number of user clusters.')
@click.option('-ic', '--item-clusters', 'n_cltr_i', type=int, required=True, help='The number of item clusters.')
@click.pass_context
def co_cluster(ctx, **params):
    model = SurpriseModel(**ctx.obj)
    model.train(surprise.CoClustering, **params)


@cli.command(help='Fit the item-based KNN algorithm.')
@click.option('-b', '--with-baseline', is_flag=True, help='Enable baseline.')
@click.option('-k', '--neighbours', 'k', type=int, required=True, help='The number of neighbours.')
@click.pass_context
def item_based(ctx, with_baseline, **params):
    model = SurpriseModel(**ctx.obj)
    model.train(**{
        'train_class': surprise.KNNWithMeans if not with_baseline else surprise.KNNBaseline,
        'sim_options': {'name': 'pearson' if not with_baseline else 'pearson_baseline', 'user_based': False},
        **params,
    })


@cli.command(help='Fit the Slope One algorithm.')
@click.pass_context
def slope_one(ctx, **params):
    model = SurpriseModel(**ctx.obj)
    model.train(surprise.SlopeOne, **params)


@cli.command(help='Fit the SVD algorithm.')
@click.option('-e', '--epochs', 'n_epochs', type=int, required=True, help='The number of epochs.')
@click.option('-f', '--factors', 'n_factors', type=int, required=True, help='The number of factors.')
@click.option('-l', '--learn-rate', 'lr_all', type=float, required=True, help='The learning rate.')
@click.option('-r', '--reg-term', 'reg_all', type=float, required=True, help='The regularization term.')
@click.pass_context
def svd(ctx, **params):
    model = SurpriseModel(**ctx.obj)
    model.train(surprise.SVD, **params)


@cli.command(help='Fit the SVD++ algorithm.', name='svd++')
@click.option('-e', '--epochs', 'n_epochs', type=int, required=True, help='The number of epochs.')
@click.option('-f', '--factors', 'n_factors', type=int, required=True, help='The number of factors.')
@click.option('-l', '--learn-rate', 'lr_all', type=float, required=True, help='The learning rate.')
@click.option('-r', '--reg-term', 'reg_all', type=float, required=True, help='The regularization term.')
@click.pass_context
def svdpp(ctx, **params):
    model = SurpriseModel(**ctx.obj)
    model.train(surprise.SVDpp, **params)


@cli.command(help='Fit the user-based KNN algorithm.')
@click.option('-b', '--with-baseline', is_flag=True, help='Enable baseline.')
@click.option('-k', '--neighbours', type=int, required=True, help='The number of neighbours.')
@click.pass_context
def user_based(ctx, with_baseline, **params):
    model = SurpriseModel(**ctx.obj)
    model.train(**{
        'train_class': surprise.KNNWithMeans if not with_baseline else surprise.KNNBaseline,
        'sim_options': {'name': 'pearson' if not with_baseline else 'pearson_baseline', 'user_based': True},
        **params,
    })


if __name__ == '__main__':
    cli(obj={})
