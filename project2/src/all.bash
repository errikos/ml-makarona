#!/usr/bin/env bash

TEMP_DIR=".tmp"
DATA_DIR="data"
OUT_DIR="out"

SPLIT_RATIO=0.9
SPLIT_SEED=988

# these are the models we are going to train and blend
declare -A MODELS=(
    ["ALS"]="als --epochs=100 --lambda=0.009 --rank=10"
    ["Co-Clustering"]="co-cluster --epochs=150 --user-clusters=2 --item-clusters=7"
    ["Item-Mean"]="item-mean"
    ["Item-based-KNN"]="item-based --neighbours=90"
    ["Item-based-KNN-baseline"]="item-based --with-baseline --neighbours=90"
    ["User-Mean"]="user-mean"
    ["User-based-KNN"]="user-based --neighbours=900"
    ["User-based-KNN-baseline"]="user-based --with-baseline --neighbours=400"
    ["Slope-One"]="slope-one"
    ["SVD"]="svd --epochs=400 --factors=10 --learn-rate=0.0002 --reg-term=0.009"
    ["SVD++"]="svd++ --epochs=400 --factors=10 --learn-rate=0.0002 --reg-term=0.009"
    ["Z-Score"]="z-score --neighbours=90"
)

# print some user-friendly info
echo "This script will run all models and then blend them together to produce a final"
echo "submission CSV file."
echo
echo "The models that will be run and blended are:"
for model in "${!MODELS[@]}"; do
    echo "  ${model}"
done
echo
echo "A temporary directory will be created in"                         \
     "${TEMP_DIR} in order to save intermediate results."
echo

# give a last chance to stop
read -n1 -r -p "Press any key to continue (or Ctrl-C to abort)... " key
echo

# create needed directories
mkdir -p "${TEMP_DIR}" "${OUT_DIR}"
mkdir -p "${TEMP_DIR}/predictions_testing"
mkdir -p "${TEMP_DIR}/predictions_submission"

# normalize the dataset (r{user}_c{item},{rating} -> {user},{item},{rating})
echo -n "> Normalizing format of train.csv... "
python3 tools.py normalize                                              \
    --input "${DATA_DIR}/train.csv"                                     \
    --output "${TEMP_DIR}/train.csv" || exit
echo "DONE"

# normalize the submission file (r{user}_c{item},{rating} -> {user},{item},{rating})
echo -n "> Normalizing format of submission.csv... "
python3 tools.py normalize                                              \
    --input "${DATA_DIR}/submission.csv"                                \
    --output "${TEMP_DIR}/submission.csv" || exit
echo "DONE"

# split the dataset into 90% training and 10% testing
echo -n "> Splitting train.csv into 90% training and 10% testing... "
python3 tools.py split-train-test "${TEMP_DIR}/train.csv"               \
    --ratio "${SPLIT_RATIO}"                                            \
    --seed "${SPLIT_SEED}" || exit
echo "DONE"

# train all models with the training dataset and make predictions on the testing dataset
echo "> Training all models and predicting for the train/test datasets:"
for model in "${!MODELS[@]}"; do
    echo -n "| Running ${model}... "
    python3 train.py                                                    \
        "${TEMP_DIR}/training.csv"                                      \
        "${TEMP_DIR}/testing.csv"                                       \
        "${TEMP_DIR}/predictions_testing/${model}_testing.csv"          \
        ${MODELS[$model]} > /dev/null || exit
    echo "DONE"
done

# train all models with the whole dataset and make predictions on the submission dataset
echo "> Training all models and predicting for the full/submission datasets:"
for model in "${!MODELS[@]}"; do
    echo -n "| Running ${model}... "
    python3 train.py                                                    \
        "${TEMP_DIR}/train.csv"                                         \
        "${TEMP_DIR}/submission.csv"                                    \
        "${TEMP_DIR}/predictions_submission/${model}_submission.csv"    \
        ${MODELS[$model]} > /dev/null || exit
    echo "DONE"
done

# do magic blending!
echo -n "> Blending the algorithms... "
python3 blend.py                                                        \
    --testing="${TEMP_DIR}/testing.csv"                                 \
    --testing-predictions="${TEMP_DIR}/predictions_testing"             \
    --submission-predictions="${TEMP_DIR}/predictions_submission"       \
    --output="${TEMP_DIR}/blended_submission.csv" || exit
echo "DONE"

echo -n "> Creating final submission file... "
python3 tools.py denormalize                                            \
    --input "${TEMP_DIR}/blended_submission.csv"                        \
    --output "${OUT_DIR}/submission.csv" || exit
echo "DONE"

# some cleanup...
echo -n "> Cleaning up... "
rm -rf ${TEMP_DIR}
echo "DONE"

# be happy
echo
echo "Success! The file to be submitted can be found in ${OUT_DIR}/submission.csv."

exit 0
