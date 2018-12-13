#!/usr/bin/env bash

PYSPARK_SCRIPT=${1}

spark-submit ${PYSPARK_SCRIPT}

echo "Combining output files..."
cat ./Spark_ALS.csv/*.csv > ./Spark_ALS_subm.csv

echo "Sorting output file..."
python3 sort_output.py ./Spark_ALS_subm.csv

echo "Cleaning up..."
rm -rf ./Spark_ALS.csv ./Spark_ALS_subm.csv
