#!/usr/bin/env python3
import os
import click
import blend
import tools


@click.command(help='Combine (blend) various model predictions, in order to obtain more accurate overall predictions.')
@click.option('-t', '--testing', 'testing_path', type=click.Path(exists=True), required=True,
              help='Path to the testing dataset, containing the real ratings.')
@click.option('-tp', '--testing-predictions', 'testing_predictions_path', type=click.Path(exists=True), required=True,
              help='Read model predictions for testing dataset from this directory.')
@click.option('-sp', '--submission-predictions', 'submission_predictions_path', type=click.Path(exists=True),
              required=True, help='Read model predictions for submission dataset from this directory.')
@click.option('-l', '--lambda', 'lambda_', type=float, default=0.001,
              help='Regularisation parameter (λ) value for ridge regression (default: 0.001).')
@click.option('-o', '--output', 'output_file', type=click.Path(exists=False), required=True,
              help='Write the final submission file to this directory.')
@click.pass_context
def main(ctx, testing_path, testing_predictions_path, submission_predictions_path, output_file, lambda_):
    ctx.invoke(blend.main,
               testing_path=testing_path,
               testing_predictions_path=testing_predictions_path,
               submission_predictions_path=submission_predictions_path,
               output_file='.blended_submission.csv',
               lambda_=lambda_)

    print()
    print('Writing output to {f}'.format(f=output_file))
    ctx.invoke(tools.denormalize, input_path='.blended_submission.csv', output_path=output_file)
    os.remove('.blended_submission.csv')


if __name__ == '__main__':
    main(obj={})
