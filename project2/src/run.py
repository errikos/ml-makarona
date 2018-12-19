#!/usr/bin/env python3
import click
import os
import sys
import pathlib
import shutil

import tools
import train
import blend


TEMP_DIR = os.path.join('.', '.tmp')
DATA_DIR = os.path.join('.', 'data')
OUT_DIR = os.path.join('.', 'out')


MODELS = {
    'ALS': {'cmd': train.als, 'args': {'epochs': 100, 'lambda_': 0.009, 'rank': 10}},
    'Co-Clustering': {'cmd': train.co_cluster, 'args': {'n_epochs': 150, 'n_cltr_u': 2, 'n_cltr_i': 7}},
    'Item-based-KNN': {'cmd': train.item_based, 'args': {'k': 90}},
    'Item-based-KNN-baseline': {'cmd': train.item_based, 'args': {'k': 90, 'with_baseline': True}},
    'Slope-One': {'cmd': train.slope_one, 'args': {}},
    'SVD': {'cmd': train.svd, 'args': {'n_epochs': 400, 'n_factors': 10, 'lr_all': 0.0002, 'reg_all': 0.009}},
    'SVD++': {'cmd': train.svdpp, 'args': {'n_epochs': 400, 'n_factors': 10, 'lr_all': 0.0002, 'reg_all': 0.009}},
    'User-based-KNN': {'cmd': train.user_based, 'args': {'k': 900}},
    'User-based-KNN-baseline': {'cmd': train.user_based, 'args': {'k': 400, 'with_baseline': True}},
}


def _create_dirs():
    pathlib.Path(TEMP_DIR).mkdir(exist_ok=True)
    pathlib.Path(os.path.join(TEMP_DIR, 'predictions_testing')).mkdir(exist_ok=True)
    pathlib.Path(os.path.join(TEMP_DIR, 'predictions_submission')).mkdir(exist_ok=True)
    pathlib.Path(OUT_DIR).mkdir(exist_ok=True)


def _prompt_to_continue():
    try:
        print()
        input('Please press ENTER to continue... ')
    except KeyboardInterrupt:
        print()
        print('Aborting...')
    print()


def _normalize_datasets(click_ctx):
    print('> Normalizing train.csv...', end=' ')
    sys.stdout.flush()
    click_ctx.invoke(tools.normalize,
                     input_path=os.path.join(DATA_DIR, 'train.csv'),
                     output_path=os.path.join(TEMP_DIR, 'train.csv'))
    print('DONE')

    print('> Normalizing submission.csv...', end=' ')
    sys.stdout.flush()
    click_ctx.invoke(tools.normalize,
                     input_path=os.path.join(DATA_DIR, 'submission.csv'),
                     output_path=os.path.join(TEMP_DIR, 'submission.csv'))
    print('DONE')


def _split_to_train_test(click_ctx):
    print('> Splitting train.csv into 90% training and 10% testing...', end=' ')
    sys.stdout.flush()
    click_ctx.invoke(tools.split_train_test,
                     dataset_path=os.path.join(TEMP_DIR, 'train.csv'),
                     ratio=0.9,
                     seed=988)
    print('DONE')


def _train_models(click_ctx, train_with, predict_for, desc):
    print('> Training all models and predicting for the {desc} dataset:'.format(desc=desc))
    for name, model in MODELS.items():
        print('| Running {m}...'.format(m=name))
        click_ctx.obj.update(train_data_path=train_with,
                             predict_data_path=predict_for,
                             output_path=os.path.join(TEMP_DIR, 'predictions_'+desc, name+'_submission.csv'))
        click_ctx.invoke(model['cmd'], **model['args'])
    print('DONE')


def _blend(click_ctx):
    print('> Blending the algorithms...')
    click_ctx.invoke(blend.main,
                     testing_path=os.path.join(TEMP_DIR, 'testing.csv'),
                     testing_predictions_path=os.path.join(TEMP_DIR, 'predictions_testing'),
                     submission_predictions_path=os.path.join(TEMP_DIR, 'predictions_submission'),
                     output_file=os.path.join(TEMP_DIR, 'blended_submission.csv'))
    print('DONE')


def _create_submission(click_ctx):
    print('> Creating final submission file...', end=' ')
    sys.stdout.flush()
    click_ctx.invoke(tools.denormalize,
                     input_path=os.path.join(TEMP_DIR, 'blended_submission.csv'),
                     output_path=os.path.join(OUT_DIR, 'submission.csv'))
    print('Done')


def _cleanup():
    shutil.rmtree(pathlib.Path(TEMP_DIR))


@click.command(help='Run the whole training and prediction pipeline.')
@click.pass_context
def main(ctx, **kwargs):
    ctx.obj.update(**kwargs)
    print('This script will run all models and then blend them together to produce\n'
          'a final submission CSV file.')
    print()
    print('The models that will be run and blended are:')
    for model_name in MODELS.keys():
        print('  {m}'.format(m=model_name))
    print()
    print('A temporary directory will be created in {tmp_dir}, in order to save the\n'
          'intermediate results.'.format(tmp_dir=TEMP_DIR))

    _prompt_to_continue()

    _create_dirs()
    _normalize_datasets(ctx)
    _split_to_train_test(ctx)

    print()
    _train_models(ctx,
                  train_with=os.path.join(TEMP_DIR, 'training.csv'),
                  predict_for=os.path.join(TEMP_DIR, 'testing.csv'),
                  desc='testing')

    print()
    _train_models(ctx,
                  train_with=os.path.join(TEMP_DIR, 'train.csv'),
                  predict_for=os.path.join(TEMP_DIR, 'submission.csv'),
                  desc='submission')

    print()
    _blend(ctx)

    print()
    _create_submission(ctx)
    # _cleanup()


if __name__ == '__main__':
    main(obj={})
