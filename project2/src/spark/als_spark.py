from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

from pyspark.context import SparkContext
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('ALS').getOrCreate()


MAX_ITER=50
LAMBDA=0.05
RANK=10


sc = SparkContext.getOrCreate()
sc.setCheckpointDir('checkpoint/')


def deal_line(row_col, rating):
    row, col = row_col.split("_")
    row = row.replace("r", "")
    col = col.replace("c", "")
    return int(row)-1, int(col)-1, float(rating)


def form_id(row_id, col_id):
    return 'r{r}_c{c}'.format(r=row_id, c=col_id)


def _clip(n, lower=1, upper=5):
    return min(5, max(1, n))


def form_pred(p):
    return _clip(int(round(p)))


def create_subm_line(r):
    user_id, pred = r
    item_id, rating = pred
    return form_id(user_id+1, item_id+1), form_pred(rating)


lines = spark.read.csv('train.csv', sep=',', header=True).rdd
parts = lines.map(lambda row: deal_line(*row))
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                     rating=float(p[2])))
ratings = spark.createDataFrame(ratingsRDD)
training, test = ratings.randomSplit([0.9, 0.1])

# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(rank=RANK, maxIter=MAX_ITER, regParam=LAMBDA, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop")
model = als.fit(training)

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)


# Generate movie recommendations for each user
userRecsRDD = (model.recommendForAllUsers(1000).rdd
                    .flatMapValues(lambda v: v)
                    .map(create_subm_line))
userRecsDF = spark.createDataFrame(userRecsRDD, schema=('Id', 'Prediction'))

# Load submission row/cols into a DataFrame
submissionRDD = (spark.read.csv('submission.csv', sep=',').rdd
                      .map(lambda row: (row[0], )))
submissionDF = spark.createDataFrame(submissionRDD, schema=('Id', ))

wantedIds = submissionDF.join(userRecsDF, on='Id')
wantedIds.sort('Id').write.csv('Spark_ALS.csv')

print("Root-mean-square error = " + str(rmse))

# Generate top 10 user recommendations for each movie
# movieRecs = model.recommendForAllItems(10)

# user = ratings.select(als.getUserCol()).filter(ratings.userId == 0)
# userRecs = model.recommendForUserSubset(user, 10)
# print(userRecs.collect())

# Generate top 10 movie recommendations for a specified set of users
# users = ratings.select(als.getUserCol()).distinct().limit(3)
# userSubsetRecs = model.recommendForUserSubset(users, 10)
# Generate top 10 user recommendations for a specified set of movies
# movies = ratings.select(als.getItemCol()).distinct().limit(3)
# movieSubSetRecs = model.recommendForItemSubset(movies, 10)
