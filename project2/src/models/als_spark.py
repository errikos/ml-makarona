#!/usr/bin/env python3
import click

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

from pyspark.context import SparkContext
from pyspark.sql import SparkSession

sc = SparkContext.getOrCreate()
sc.setLogLevel('ERROR')
sc.setCheckpointDir('checkpoint/')

spark = SparkSession.builder.appName('ALS').getOrCreate()


SEED = 4
EPOCHS = 100
LAMBDA = 0.009
RANK = 10


def train(dataset, rank=RANK, epochs=EPOCHS, lambda_=LAMBDA, seed=SEED):
    """Build the recommendation model using ALS on the training dataset."""
    als = ALS(rank=rank, maxIter=epochs, regParam=lambda_, seed=seed,
              userCol='user', itemCol='item', ratingCol='rating',
              coldStartStrategy='drop')
    model = als.fit(dataset)
    return model


def flatten(t):
    user, prediction = t
    item, rating = prediction
    return user, item, rating


def predict(model, predict_items, collect=True):
    wanted_user_item_pairs = predict_items.drop('rating')
    predictions = spark.createDataFrame(
                    model.recommendForAllUsers(1000).rdd
                       .flatMapValues(lambda v: v)
                       .map(flatten),
                    schema=('user', 'item', 'rating'))
    wanted_predictions = predictions.join(wanted_user_item_pairs, on=['user', 'item'])
    return wanted_predictions if not collect else wanted_predictions.collect()


def evaluate(model, testing):
    """Evaluate the model by computing the RMSE on the testing dataset."""
    predictions = model.transform(testing)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    return evaluator.evaluate(predictions)


def split_train_test(dataset, ratio):
    return dataset.randomSplit([ratio, 1.0-ratio])


def load_ratings(path):
    """Load the (user, item, rating) triplets into a DataFrame."""
    return spark.read.csv(path, sep=',', header=True, schema='user INT, item INT, rating FLOAT')


@click.command()
@click.argument('input_file', type=click.Path(exists=True), required=True)
@click.argument('submission_file', type=click.Path(exists=True), required=True)
@click.argument('output_path', type=click.Path(exists=False), required=True)
@click.pass_context
def main(ctx, input_file, submission_file, output_path, **kwargs):
    ctx.obj.update(**kwargs)

    ratings = load_ratings(input_file)
    submission = load_ratings(submission_file)

    model = train(ratings)
    predictions = predict(model, submission, collect=False)

    predictions.write_csv(output_path, header=False)


if __name__ == '__main__':
    main(obj={})
