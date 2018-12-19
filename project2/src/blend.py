#!/usr/bin/env python3
import click
import glob
import itertools
import os

import numpy as np
import helpers


def _list_files(path):
    return glob.glob(os.path.join(path, '*.csv'))


def _print_files(files):
    print('Found {n} CSV files in {d}:'.format(n=len(files), d=os.path.dirname(files[0])))
    for f in files:
        print('  -', os.path.basename(f))
    print()


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations."""
    n, d = tx.shape
    lambda_ = 2 * n * lambda_

    a = tx.T.dot(tx) + lambda_ * np.eye(d)
    b = tx.T.dot(y)

    return np.linalg.solve(a, b)


def compute_rmse(real, prediction):
    t = real - np.round(prediction)
    N, = t.shape
    return np.sqrt(t.dot(t) / N)


def load_ratings(path):
    get_rating = lambda t: float(t.strip().split(',')[2])
    return np.fromiter(map(get_rating, helpers.read_lines(path, header=False)), dtype=np.float)


def model_combinations(models, min_k=1):
    for i in range(min_k, len(models)+1):
        for combination in itertools.combinations(models.items(), i):
            yield combination


def evaluate_combination(combination, real_ratings, lambda_):
    model_names, model_predictions = list(zip(*combination))
    y = np.array(real_ratings)
    tx = np.array(model_predictions).transpose()
    # compute weights and RMSE
    w = ridge_regression(y, tx, lambda_)
    rmse = compute_rmse(y, tx.dot(w))
    return model_names, w, rmse


def get_submission_id_pairs(submission_prediction_files):
    get_user_item_pair = lambda t: t.strip().split(',')[:2]
    f = submission_prediction_files[0]
    return map(get_user_item_pair, helpers.read_lines(f, header=False))


def make_weighted_predictions(submission_ratings, comb_names, w):
    model_names, model_predictions = list(zip(*[(name, submission_ratings[name]) for name in comb_names]))
    X = np.array(model_predictions).transpose()
    return X.dot(w)


def blend(testing_dataset_path, testing_prediction_files, submission_prediction_files, output_file, lambda_):
    get_model_name = lambda p: os.path.splitext(os.path.basename(p))[0]
    # load the real ratings from the testing dataset
    testing_ratings = load_ratings(testing_dataset_path)
    # load the predictions from the testing and submission prediction files
    testing_predictions = dict(zip(map(get_model_name, testing_prediction_files),
                               map(load_ratings, testing_prediction_files)))
    submission_predictions = dict(zip(map(get_model_name, submission_prediction_files),
                                  map(load_ratings, submission_prediction_files)))
    # evaluate each combination of the model predictions w.r.t. the testing dataset
    blended_ratings = [evaluate_combination(c, testing_ratings, lambda_)
                       for c in model_combinations(testing_predictions)]
    # reveal optimal combination and its weight vector
    opt_comb_names, opt_w, opt_rmse = min(blended_ratings, key=lambda t: t[2])

    print('Optimal blending is:')
    spec_len = max(map(len, opt_comb_names))
    spec = '  - {model:%d} x {w}' % spec_len
    for model, w in zip(opt_comb_names, opt_w):
        print(spec.format(model=model, w=w))
    print('With RMSE:', opt_rmse)

    # prepare submission user/item pairs and get weighted predictions
    submission_user_item_pairs = get_submission_id_pairs(submission_prediction_files)
    weighted_submission_predictions = make_weighted_predictions(submission_predictions, opt_comb_names, opt_w)

    helpers.write_normalized(output_file, ((u, i, int(round(r)))
                                           for (u, i), r in zip(submission_user_item_pairs,
                                                                weighted_submission_predictions)))


@click.command(help='Combine (blend) various model predictions, in order to obtain more accurate overall predictions.')
@click.option('-t', '--testing', 'testing_path', type=click.Path(exists=True), required=True,
              help='Path to the testing dataset, containing the real ratings.')
@click.option('-tp', '--testing-predictions', 'testing_predictions_path', type=click.Path(exists=True), required=True,
              help='Read model predictions for testing dataset from this directory.')
@click.option('-sp', '--submission-predictions', 'submission_predictions_path', type=click.Path(exists=True),
              required=True, help='Read model predictions for submission dataset from this directory.')
@click.option('-l', '--lambda', 'lambda_', type=float, default=0.001,
              help='Regularisation parameter (λ) value for ridge regression (default: 0.0007).')
@click.option('-o', '--output', 'output_file', type=click.Path(exists=False), required=True,
              help='Write the blended submission file to this directory.')
@click.pass_context
def main(ctx, testing_path, testing_predictions_path, submission_predictions_path, output_file, lambda_, **kwargs):
    ctx.obj.update(**kwargs)
    testing_prediction_files = _list_files(testing_predictions_path)
    submission_prediction_files = _list_files(submission_predictions_path)
    if not testing_prediction_files:
        click.echo('No CSV files found in testing predictions directory!')
        raise click.Abort()
    if not submission_prediction_files:
        click.echo('No CSV files found in submission predictions directory!')
        raise click.Abort()
    if len(testing_prediction_files) != len(submission_prediction_files):
        click.echo('The prediction directories must have the same number of files.')
        raise click.Abort()
    _print_files(testing_prediction_files)
    _print_files(submission_prediction_files)

    input('Please press ENTER to continue... ')
    blend(testing_path, testing_prediction_files, submission_prediction_files, output_file, lambda_)


if __name__ == '__main__':
    main(obj={})
