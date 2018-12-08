import numpy as np
import csv

from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV

from helpers import load_data

def test():

	# Load ratings
	ratings_path = "../../data/train_clean.csv"
	reader = Reader(line_format='item user rating', sep=',', skip_lines=1)
	ratings = Dataset.load_from_file(ratings_path, reader)

	# Build SVD model.
	algorithm = SVD(n_epochs=30, lr_all=0.001, reg_all=0.001)

	# Run 3-fold cross-validation and print results
	cross_validate(algorithm, ratings, \
					measures=['RMSE'], cv=3, verbose=True)


def submit():
	# Load ratings
	ratings_path = "../../data/train_clean.csv"
	reader = Reader(line_format='item user rating', sep=',', skip_lines=1)
	ratings = Dataset.load_from_file(ratings_path, reader)

	# Retrieve the trainset.
	train_ratings = ratings.build_full_trainset()

	# Build KNN user based model and train it.
	sim_options = {'name': 'pearson', 'user_based': True}
	algorithm = SVD(n_epochs=30, lr_all=0.001, reg_all=0.001)
	algorithm.fit(train_ratings)

	# Get submission file format
	print("Producing submission file...")
	sample_submission_path = "../../data/submission.csv"
	test_ratings = load_data(sample_submission_path, sparse_matrix=False)

	rows, cols = np.nonzero(test_ratings)
	zp = list(zip(rows, cols))
	zp.sort(key = lambda tup: tup[1])

	# Create submission file
	submission_path = "./submissions/surprise_svd.csv"
	csvfile = open(submission_path, 'w')

	fieldnames = ['Id', 'Prediction']
	writer = csv.DictWriter(csvfile, delimiter=",", \
						fieldnames=fieldnames, lineterminator = '\n')
	writer.writeheader()

	counter = 0
	for row, col in zp:

		counter += 1
		if counter % 1000 == 0:
			print("Progress: %d/%d" % (counter, len(rows)))

		iid = str(row)
		uid = str(col)
		val = int(round(algorithm.predict(uid, iid)[3]))

		if val > 5:
			val = 5
		elif val < 1:
			val = 1
		
		r = "r" + str(row + 1)
		c = "c" + str(col + 1)
		writer.writerow({'Id': r + "_" + c, 'Prediction': val})


if __name__ == '__main__':
	
	# test()
	submit()