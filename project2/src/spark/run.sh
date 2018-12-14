#!/usr/bin/env bash
SPARK_BIN="/Users/ergys/Downloads/spark-2.4.0/bin"
PYSPARK_SCRIPT=${1}

${SPARK_BIN}/spark-submit ${PYSPARK_SCRIPT}
if [ $? -eq 0 ]; then
    echo "Combining output files..."
    cat ./Spark_ALS.csv/*.csv > ./Spark_ALS_subm.csv

    echo "Sorting output file..."
    python3 sort_output.py ./Spark_ALS_subm.csv

    echo "Cleaning up..."
    rm -rf ./Spark_ALS.csv ./Spark_ALS_subm.csv
else
    echo "${0}: Spark invocation failed."
fi

