import numpy as np
import csv

from surprise import KNNWithMeans
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV

from helpers import load_data 

def tune():

	# Load ratings
	ratings_path = "../../data/train_clean.csv"
	reader = Reader(line_format='item user rating', sep=',', skip_lines=1)
	ratings = Dataset.load_from_file(ratings_path, reader)

	param_grid = {'k': [50, 100, 200, 300, 500],
				  'sim_options': {'name': ['pearson'],
			  					  'user_based': [True]
			  					  }
				 }

	gs = GridSearchCV(KNNWithMeans, param_grid, \
					  measures=['rmse'], cv=3, n_jobs=2)

	gs.fit(ratings)

	# Best RMSE score
	print("Best RMSE score:", gs.best_score['rmse'])

	# Combination of parameters that gave the best RMSE score
	print("With params:", gs.best_params['rmse'])


def test():

	# Load ratings
	ratings_path = "../../data/train_clean.csv"
	reader = Reader(line_format='item user rating', sep=',', skip_lines=1)
	ratings = Dataset.load_from_file(ratings_path, reader)

	# Build KNN user based model.
	sim_options = {'name': 'pearson', 'user_based': True}
	algorithm = KNNWithMeans(k=50, sim_options=sim_options)

	# Run 5-fold cross-validation and print results
	cross_validate(algorithm, ratings, \
					measures=['RMSE'], cv=5, verbose=True)


def submit(K):
	# Load ratings
	ratings_path = "../../data/train_clean.csv"
	reader = Reader(line_format='item user rating', sep=',', skip_lines=1)
	ratings = Dataset.load_from_file(ratings_path, reader)

	# Retrieve the trainset.
	train_ratings = ratings.build_full_trainset()

	# Build KNN user based model and train it.
	sim_options = {'name': 'pearson', 'user_based': True}
	algorithm = KNNWithMeans(k=K, sim_options=sim_options)
	algorithm.fit(train_ratings)

	# Get submission file format
	print("Producing submission file...")
	sample_submission_path = "../../data/submission.csv"
	test_ratings = load_data(sample_submission_path, sparse_matrix=False)

	rows, cols = np.nonzero(test_ratings)
	zp = list(zip(rows, cols))
	zp.sort(key = lambda tup: tup[1])

	# Create submission file
	submission_path = "./submissions/surprise_user_based_top" + str(K) +".csv"
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
	
	# tune()
	# test()
	submit(200)