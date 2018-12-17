import os

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

from pyspark.context import SparkContext
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('ALS').getOrCreate()


SEED = 4
EPOCHS = 50
LAMBDA = 0.09
RANK = 10
TRAINING_RATIO = 1.0

INPUT_FILE = os.path.join('988_train_normalized.csv')
SUBMISSION_FILE = os.path.join('988_test_normalized.csv')


sc = SparkContext.getOrCreate()
sc.setLogLevel('ERROR')
sc.setCheckpointDir('checkpoint/')


def die(code=0):
    import sys
    sys.exit(code)


def deal_line(row_col, rating):
    row, col = row_col.split("_")
    row = row.replace("r", "")
    col = col.replace("c", "")
    return int(row)-1, int(col)-1, float(rating)


def form_id(row_id, col_id):
    return 'r{r}_c{c}'.format(r=row_id, c=col_id)


def _clip(n, lower=1, upper=5):
    return min(upper, max(lower, n))


def form_pred(p):
    return p


def create_subm_line(r):
    user_id, pred = r
    item_id, rating = pred
    return form_id(user_id, item_id), form_pred(rating)


def load_ratings(path):
    """Load the (userId, movieId, rating) triplets into a DataFrame."""
    lines = spark.read.csv(path, sep=',', header=True).rdd
    parts = lines.map(lambda row: deal_line(*row))
    ratings = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]), rating=float(p[2])))
    return spark.createDataFrame(ratings)


def train(dataset, rank=RANK, epochs=EPOCHS, lambda_=LAMBDA, seed=SEED):
    """Build the recommendation model using ALS on the training dataset."""
    als = ALS(rank=rank, maxIter=epochs, regParam=lambda_, seed=seed,
              userCol="userId", itemCol="movieId", ratingCol="rating",
              coldStartStrategy="drop")
    model = als.fit(dataset)
    return model


def evaluate(model, testing):
    """Evaluate the model by computing the RMSE on the testing dataset."""
    predictions = model.transform(testing)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    return evaluator.evaluate(predictions)


def generate_recommendations(model):
    """Generate recommendations for all users."""
    user_recs = (model.recommendForAllUsers(1000).rdd
                 .flatMapValues(lambda v: v)
                 .map(create_subm_line))
    return spark.createDataFrame(user_recs, schema=('Id', 'Prediction'))


def split_train_test(dataset, ratio):
    # training based on the whole dataset (no testing)
    if not 1.0-ratio:
        model = train(dataset)
        print('recommend: no training/testing split; cannot report RMSE')
        return generate_recommendations(model)
    # split the dataset into training and testing
    return dataset.randomSplit([ratio, 1.0-ratio])


def recommend(dataset):
    training, testing = split_train_test(dataset, ratio=TRAINING_RATIO)
    # train the ALS model
    model = train(training)
    # evaluate the model by computing the RMSE
    rmse = evaluate(model, testing)
    print('recommend: Root-mean-square error = {rmse}'.format(rmse=rmse))
    # generate recommendations (ratings) for all users
    return generate_recommendations(model)


def create_submission(recommendations, submission_file):
    """Create the submission CSV based on recommendations."""
    # load submission row/cols into a DataFrame
    submission = spark.createDataFrame(spark.read.csv(submission_file, sep=',').rdd.map(lambda row: (row[0], )),
                                       schema=('Id', ))
    return submission.join(recommendations, on='Id')


def export_submission(submission, path):
    submission.write.csv(path, header=False)


def als_spark(input_file, submission_file):
    ratings = load_ratings(input_file)
    # train with ALS
    recommendations = recommend(ratings)
    # create submission DataFrame
    submission = create_submission(recommendations, submission_file)
    # export submission to file
    export_submission(submission, 'Spark_ALS.csv')


if __name__ == '__main__':
    als_spark(INPUT_FILE, SUBMISSION_FILE)
