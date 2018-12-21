#!/usr/bin/env python3
import click
import os

import helpers


def _write_normalized(data, base_path, fname):
    with open(os.path.join(base_path, fname), 'w+') as f:
        f.write('user,item,rating\n')
        f.writelines('{u},{i},{r}\n'.format(u=u, i=i, r=r) for u, i, r in data)


@click.group(help='Various useful tools for the recommendation system training workflow.')
@click.pass_context
def cli(ctx, **kwargs):
    ctx.obj.update(**kwargs)


@cli.command(help='Convert a dataset from the upstream to the normalized format.')
@click.option('-i', '--input', 'input_path', type=click.Path(exists=True), required=True,
              help='The input dataset file path.')
@click.option('-o', '--output', 'output_path', type=click.Path(exists=False), required=True,
              help='The output dataset file path.')
def normalize(input_path, output_path, **kwargs):
    helpers.normalize(input_path, output_path)


@cli.command(help='Convert a dataset from the normalized to the upstream format.')
@click.option('-i', '--input', 'input_path', type=click.Path(exists=True), required=True,
              help='The input dataset file path.')
@click.option('-o', '--output', 'output_path', type=click.Path(exists=False), required=True,
              help='The output dataset file path.')
def denormalize(input_path, output_path, **kwargs):
    helpers.denormalize(input_path, output_path)


@cli.command(help='Split a normalized dataset into training and testing.')
@click.argument('dataset_path', type=click.Path(exists=True), required=True)
@click.option('-r', '--ratio', metavar='RATIO', type=float, default=0.9, help='The split ratio (default: 0.9).')
@click.option('-s', '--seed', metavar='SEED', type=int, default=988, help='The seed to use (default: 988).')
def split_train_test(dataset_path, **kwargs):
    train, test = helpers.split_normalized_data(dataset_path, **kwargs)
    _write_normalized(train, os.path.dirname(dataset_path), 'training.csv')
    _write_normalized(test, os.path.dirname(dataset_path), 'testing.csv')


if __name__ == '__main__':
    cli(obj={})
